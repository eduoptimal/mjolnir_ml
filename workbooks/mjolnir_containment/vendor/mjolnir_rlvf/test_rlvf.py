"""Tests for mjolnir_rlvf — the product RLVF pipeline.

Everything here is offline: the reward tables are pure, probe checks the
authority against the suite's authored oracle, and the pipeline test runs
rollout -> pairs -> eval with a CANNED proposer (reference solutions), so the
wiring is proven without a model or a network. The authority is imported
read-only — the same boundary the package documents.
"""
from __future__ import annotations

import json
import os
import pytest

from mjolnir_rlvf import cli, config, reference, reward, tasks


def test_params_parsing(tmp_path):
    assert cli.parse_params("") == {}
    assert cli.parse_params('{"k": 2}') == {"k": 2}
    f = tmp_path / "p.json"
    f.write_text('{"suite": "A"}')
    assert cli.parse_params(f"@{f}") == {"suite": "A"}
    with pytest.raises(ValueError):
        cli.parse_params("{not json")
    with pytest.raises(ValueError):
        cli.parse_params("[1,2]")


def test_reward_tables_are_the_measured_ones():
    # the mandate: honest abstention positive on undecidable, overclaim worst
    assert reward.calibration_reward("certified", "decidable") == 1.0
    assert reward.calibration_reward("abstain", "undecidable") == 0.75
    assert reward.calibration_reward("abstain", "decidable") == 0.0
    # P2/P3: fabricated and premise invention never beat honest abstention
    assert reward.calibration_reward("fabricated", "decidable") == -1.0
    assert reward.calibration_reward("premise_invention", "undecidable") == -1.0
    assert reward.calibration_reward("refuted", "decidable") == -1.0
    # typed: a theorem proof outranks a declared premise
    assert reward.typed_reward("theorem", "CERTIFIED") == 1.0
    assert reward.typed_reward("premise", "CERTIFIED") == 0.5
    assert reward.typed_reward("theorem", "UNVERIFIABLE") == 0.0


def test_probe_stage_is_offline_and_green():
    """probe = single-writer + the authored oracle against the authority:
    the rung that must pass before any model is contacted."""
    assert cli.stage_probe() == 0


def test_pipeline_offline_with_a_canned_proposer(monkeypatch, tmp_path):
    from mjolnir_rlvf import proposer
    monkeypatch.setattr(config, "DATA", str(tmp_path / "data"))
    refs = {r["task"]["id"]: r["code"] for r in reference.probe_rows()
            if r["code"]}
    by_fn = {t["fn"]: t for t in tasks.load_all()}

    def canned(model, prompt, temperature, seed, num_predict=None,
               backend=None, adapter=None):
        # K=2 with per-sample contrast: even seed -> the reference solution
        # (certified), odd seed -> an honest decline, so the within-task
        # contrast pairs.build requires actually exists. Real models create
        # this spread naturally; the canned one must not need luck for it.
        import re as _re
        if seed % 2:
            return "ABSTAIN: negative contrast sample"
        m = _re.search(r"FUNCTION: (\w+)\(", prompt)
        task = by_fn.get(m.group(1)) if m else None
        code = refs.get(task["id"]) if task else None
        return code or "ABSTAIN: no reference for this task"

    monkeypatch.setattr(proposer, "generate", canned)
    run_id = cli.stage_rollout("canned-model",
                               {"suite": "all", "k": 2, "limit": 6})
    run_dir = os.path.join(str(tmp_path / "data"), "runs", run_id)
    rows = [json.loads(l) for l in
            open(os.path.join(run_dir, "rollouts.jsonl"))]
    assert len(rows) == 12
    # every row carries the authority's typed rewards + run provenance
    assert all("reward_calibration" in r and "reward_typed" in r
               for r in rows)
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    assert meta["authority_fingerprint"] and meta["benchmark_fingerprint"]

    cli.stage_pairs({}, run_id)
    train = os.path.join(run_dir, "pairs", "train.jsonl")
    valid = os.path.join(run_dir, "pairs", "valid.jsonl")
    allp = [json.loads(l) for l in open(os.path.join(run_dir, "pairs.jsonl"))]
    assert len(allp) > 0, "within-task contrast produced no pairs"
    assert os.path.isfile(train) and os.path.isfile(valid)
    assert json.loads(open(train).readline()).keys() >= {
        "prompt", "chosen", "rejected"}

    cli.stage_eval({}, run_id)
    ev = json.load(open(os.path.join(run_dir, "eval.json")))
    assert ev["summary"]["n"] == 12 and ev["run_id"] == run_id


def test_bad_stage_and_missing_run_exit_2():
    with pytest.raises(SystemExit) as e:
        cli.run("m", "{}", "bogus")
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.run("m", "{}", "pairs")   # no runs anywhere
    assert e.value.code == 2


def test_serve_argv_mapping_is_pure_and_complete():
    a = cli._serve_argv({}, None)
    assert "--backend" in a and "--port" in a and "--model-name" in a
    a2 = cli._serve_argv({"serve_backend": "ollama", "port": 9000},
                         "qwen2.5-coder:1.5b-instruct")
    assert a2[a2.index("--ollama-model") + 1] == "qwen2.5-coder:1.5b-instruct"
    assert a2[a2.index("--port") + 1] == "9000"
    a3 = cli._serve_argv(
        {"serve_backend": "mlx", "model_path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
         "adapter": "/tmp/adapter"}, None)
    assert "Qwen/Qwen2.5-Coder-1.5B-Instruct" in a3
    assert a3[a3.index("--adapter") + 1] == "/tmp/adapter"
    a4 = cli._serve_argv({"inject_system": False}, None)
    assert "--no-inject-system" in a4


def test_serve_speaks_openai_and_injects_the_trained_contract():
    """The endpoint kilocode/VSCode point at: /health, /v1/models, and
    /v1/chat/completions in OpenAI shape — with the RLVF system prompt
    injected when the caller sends none (client's own system still wins)."""
    import threading
    import urllib.request
    from mjolnir_rlvf import serve

    class Fake:
        last = None

        def info(self):
            return {"backend": "fake", "weights": "test-only"}

        def chat(self, messages, temperature, max_tokens, seed=0):
            Fake.last = messages
            return ("def f():\n    result = 1\n    assert result == 1",
                    {"prompt_tokens": 1, "completion_tokens": 2,
                     "total_tokens": 3})

    serve.STATE.clear()
    serve.STATE.update(
        backend=Fake(), model_name="rlvf-test", created=1, system=None,
        prompt_mode="rlvf", inject_system=True, temperature=0.0,
        max_tokens=64, seed=0)
    httpd = serve.ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        health = json.load(urllib.request.urlopen(base + "/health"))
        assert health["status"] == "ok" and health["backend"] == "fake"
        models = json.load(urllib.request.urlopen(base + "/v1/models"))
        assert models["data"][0]["id"] == "rlvf-test"

        def chat(messages):
            req = urllib.request.Request(
                base + "/v1/chat/completions",
                data=json.dumps({"messages": messages}).encode(),
                headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req))

        resp = chat([{"role": "user", "content": "implement it"}])
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["content"].startswith("def f")
        assert resp["usage"]["total_tokens"] == 3
        # no client system -> the trained contract is injected
        assert Fake.last[0]["role"] == "system"
        assert "ABSTAIN" in Fake.last[0]["content"]
        # ALWAYS ON: a client's own system text is preserved AFTER the
        # contract, in ONE system message — the caller cannot deactivate
        # prompt-activated behaviour
        chat([{"role": "system", "content": "mine"},
              {"role": "user", "content": "hi"}])
        body = Fake.last[0]["content"]
        assert sum(1 for m in Fake.last if m["role"] == "system") == 1
        assert "ABSTAIN" in body and body.index("ABSTAIN") < body.index("mine")
    finally:
        httpd.shutdown()
        httpd.server_close()
        serve.STATE.clear()


def test_the_system_prompt_contract_is_pinned():
    """Prompt-activated behaviour: the serve default must BE the contract the
    model was calibrated on — assert-or-ABSTAIN, never invent a precondition."""
    from mjolnir_rlvf import prompts
    assert prompts.system_prompt("rlvf") == prompts.RLVF_SYSTEM
    assert prompts.system_prompt("agent") == prompts.AGENT_SYSTEM
    for text in (prompts.RLVF_SYSTEM, prompts.AGENT_SYSTEM):
        assert "ABSTAIN" in text and "Never" in text and "assert" in text
    assert "Never invent a precondition" in prompts.RLVF_SYSTEM
    assert "Never fabricate preconditions" in prompts.AGENT_SYSTEM


def test_rlvf_debug_switch_validation():
    """--rlvf active|inactive is a debug switch: a bad value is a usage error
    (exit 2) before any stage runs; inactive maps to an explicit
    inject_system=False in the serve params."""
    with pytest.raises(SystemExit) as e:
        cli.run("m", "{}", "probe", serve=True, rlvf="sometimes")
    assert e.value.code == 2
    # the argv mapping honours the off state (pure function)
    a = cli._serve_argv({"inject_system": False}, None)
    assert "--no-inject-system" in a
    b = cli._serve_argv({"inject_system": True}, None)
    assert "--no-inject-system" not in b
