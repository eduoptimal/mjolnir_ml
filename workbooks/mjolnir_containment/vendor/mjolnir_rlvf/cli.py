"""mjolnir_rlvf.cli — the `mjolnir rlvf` stage machine.

    mjolnir rlvf --model=<model_id> --params='{...}' [--stage <stages>]

Stages (comma-separated, or `all` = probe,rollout,pairs,train,eval):

    probe    offline: single-writer invariant + the oracle against the
             authority (reference solutions must dispose as labeled) — no model
    rollout  generate K samples per task -> dispose via the authority ->
             rollouts.jsonl (typed rewards per row; FAULT stops the line)
    pairs    typed preference pairs -> pairs/{train,valid}.jsonl (DPO layout)
    train    DPO on the pairs (needs extra `rlvf`: torch/transformers/peft/trl)
    eval     MJ-Eval ledger: cert yield, overclaim, calibration reward,
             Wilson CIs — precision/recall as the metric pair
    serve    OpenAI-compatible HTTP server for the trained model
             (VSCode / kilocode / any OpenAI client): --serve

Params JSON keys (all optional): suite, k, temperature, seed, limit, split,
run_id, backend (ollama|openai|mlx), adapter, valid_frac, model_path,
out_dir, train{epochs,seed,lr,beta,max_length,...}. `--params @file.json`
reads a file. Artifacts: ./rlvf_data/runs/<run_id>/ (MJ_RLVF_DATA overrides).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

from . import bench, config, pairs, reference, rollout, store, tasks

STAGES = ("probe", "rollout", "pairs", "train", "eval", "serve")

_MISSING_TRAIN = ('rlvf train needs the training extra: '
                  'pip install "mjolnir[rlvf]"')


def parse_params(text: str) -> dict:
    s = (text or "").strip()
    if s.startswith("@"):
        with open(s[1:], encoding="utf-8") as fh:
            s = fh.read().strip()
    if not s:
        return {}
    try:
        p = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"--params must be JSON (or @file.json): {e}") from None
    if not isinstance(p, dict):
        raise ValueError("--params must be a JSON object {…}")
    return p


def _tasks_for(p: dict) -> list:
    suite = p.get("suite", "all")
    tl = tasks.load_all() if suite in ("all", "ALL") else tasks.load_suite(suite)
    split = p.get("split")
    if split:
        keep = tasks.ids_in_split(split)
        tl = [t for t in tl if t["id"] in keep]
    limit = int(p.get("limit", 0) or 0)
    return tl[:limit] if limit else tl


def _latest_run() -> str | None:
    root = os.path.join(config.DATA, "runs")
    if not os.path.isdir(root):
        return None
    runs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    return max(runs, key=lambda d: os.path.getmtime(os.path.join(root, d))) \
        if runs else None


def stage_probe() -> int:
    """Offline gate: the authority is sane and the suite's oracle is right."""
    from mjolnir_prover.lib import certify
    ok, offenders = certify.check_single_writer()
    print(f"single_writer invariant: {ok}"
          + ("" if ok else f" offenders={offenders}"))
    mismatches = 0
    for row in reference.probe_rows():
        task, code = row["task"], row["code"]
        if code is None:
            continue
        cert, _pf, _vl = certify.certify_unit(task["fn"], code)
        status = cert.get("status")
        want = task["oracle"] == "decidable"
        good = (status == "CERTIFIED") if want else (status != "CERTIFIED")
        mismatches += 0 if good else 1
        print(f"{task['id']:22s} {task['oracle']:12s} {status:12s} "
              f"{'OK' if good else 'MISMATCH'}")
    print(f"oracle mismatches: {mismatches}")
    return 0 if (ok and mismatches == 0) else 1


def stage_rollout(model: str, p: dict) -> str:
    config.TEMPERATURE = float(p.get("temperature", config.TEMPERATURE))
    config.SEED = int(p.get("seed", config.SEED))
    k = int(p.get("k", config.DEFAULT_K))
    backend = p.get("backend")
    adapter = p.get("adapter")
    run_id = p.get("run_id") or (
        f"rlvf-{re.sub(r'[^A-Za-z0-9_.-]+', '_', model)}-"
        f"{time.strftime('%Y%m%d-%H%M%S')}")
    tl = _tasks_for(p)
    print(f"rollout: {len(tl)} task(s) x K={k} model={model} "
          f"backend={backend or config.BACKEND} run_id={run_id}")
    rollout.rollout(tl, model, k, run_id, backend=backend, adapter=adapter)
    return run_id


def stage_pairs(p: dict, run_id: str) -> None:
    path, rows = pairs.write_pairs(
        run_id, valid_frac=float(p.get("valid_frac", 0.1)))
    print(f"pairs: {len(rows)} -> {path} (+ pairs/{{train,valid}}.jsonl)")


def stage_train(p: dict, run_id: str) -> None:
    from .train import train
    t = p.get("train") or {}
    data_dir = os.path.join(store.run_dir(run_id), "pairs")
    model_path = p.get("model_path") or config.MLX_MODEL
    out_dir = p.get("out_dir") or os.path.join(store.run_dir(run_id), "dpo")
    try:
        train(data_dir, model_path, out_dir,
              epochs=float(t.get("epochs", 2.0)),
              seed=int(t.get("seed", 42)), lr=float(t.get("lr", 5e-6)),
              beta=float(t.get("beta", 0.1)),
              max_length=int(t.get("max_length", 512)))
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(2) from e


def stage_eval(p: dict, run_id: str) -> None:
    rows = store.read_jsonl(os.path.join(store.run_dir(run_id),
                                         "rollouts.jsonl"))
    summary = bench.summarize(rows)
    store.write_json(os.path.join(store.run_dir(run_id), "eval.json"),
                     {"run_id": run_id, "summary": summary})
    print(bench.render(summary))
    print(f"artifacts: {store.run_dir(run_id)}")


def _serve_argv(p: dict, model: str | None) -> list:
    """--params/--model -> argv for serve.main (pure: testable offline)."""
    b = p.get("serve_backend") or config.SERVE_BACKEND
    argv = ["--backend", b]
    if b == "ollama":
        argv += ["--ollama-model",
                 model or p.get("ollama_model") or config.SERVE_OLLAMA_MODEL]
        if p.get("ollama_url"):
            argv += ["--ollama-url", str(p["ollama_url"])]
    else:  # mlx: a model id/path for local weights + the trained adapter
        mp = p.get("model_path") or (
            model if model and ("/" in model or os.path.isdir(model))
            else config.SERVE_BASE_MODEL)
        argv += ["--model-path", mp]
        ad = p.get("adapter") or config.SERVE_ADAPTER
        if ad:
            argv += ["--adapter", ad]
    argv += ["--host", str(p.get("host", config.SERVE_HOST)),
             "--port", str(int(p.get("port", config.SERVE_PORT))),
             "--model-name", str(p.get("model_name", config.SERVED_NAME)),
             "--prompt-mode", str(p.get("prompt_mode", "rlvf")),
             "--temperature", str(float(p.get("serve_temperature",
                                              config.SERVE_TEMPERATURE))),
             "--max-tokens", str(int(p.get("max_tokens",
                                           config.SERVE_MAX_TOKENS))),
             "--seed", str(int(p.get("serve_seed", config.SERVE_SEED)))]
    if p.get("system") is not None:
        argv += ["--system", str(p["system"])]
    if p.get("inject_system") is False or p.get("no_inject_system"):
        argv += ["--no-inject-system"]
    return argv


def stage_serve(p: dict, model: str | None) -> None:
    """Block: serve the model until Ctrl-C (the trained contract travels
    with every request via the injected RLVF system prompt)."""
    from . import serve
    argv = _serve_argv(p, model)
    print("mjolnir rlvf --serve: " + " ".join(argv))
    serve.main(argv)


def run(model: str | None, params: str, stage: str,
        serve: bool = False, rlvf: str = "active") -> None:
    """Entry for `mjolnir rlvf` — resolves stages, threads the run_id,
    raises SystemExit with the product's exit code (0 ok, 1 findings-ish,
    2 missing environment/usage)."""
    try:
        p = parse_params(params)
    except ValueError as e:
        print(f"rlvf: {e}", file=sys.stderr)
        raise SystemExit(2) from None
    model = model or p.get("model") or config.DEFAULT_MODEL
    if rlvf not in ("active", "inactive"):
        print(f"rlvf: --rlvf must be active|inactive (got {rlvf!r})",
              file=sys.stderr)
        raise SystemExit(2)
    wanted = [s.strip() for s in (stage or "all").split(",") if s.strip()]
    if serve:
        wanted = ["serve"]          # --serve wins: a server is the whole job
        # the debug switch: contract ALWAYS (active) or NEVER (inactive) —
        # mapped before stage_serve so _serve_argv sees an explicit value.
        p["inject_system"] = (rlvf == "active")
    if wanted == ["all"]:
        wanted = [s for s in STAGES if s != "serve"]  # all never blocks
    bad = [s for s in wanted if s not in STAGES]
    if bad:
        print(f"rlvf: unknown stage(s) {bad} — choose from {STAGES} or 'all'",
              file=sys.stderr)
        raise SystemExit(2)

    run_id = p.get("run_id")
    code = 0
    for s in wanted:
        if s == "probe":
            code = stage_probe()
            if code:
                raise SystemExit(code)
        elif s == "serve":
            stage_serve(p, model)   # blocks; Ctrl-C returns here
        elif s == "rollout":
            run_id = stage_rollout(model, p)
            p["run_id"] = run_id
        elif s in ("pairs", "train", "eval"):
            rid = run_id or p.get("run_id") or _latest_run()
            if not rid:
                print(f"rlvf {s}: no run_id and no runs under "
                      f"{os.path.join(config.DATA, 'runs')} — run rollout first",
                      file=sys.stderr)
                raise SystemExit(2)
            if s == "pairs":
                stage_pairs(p, rid)
            elif s == "train":
                stage_train(p, rid)
            else:
                stage_eval(p, rid)
    raise SystemExit(code)
