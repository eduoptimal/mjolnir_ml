"""mjolnir_decide.model — the decision stack behind the noul answers.

Self-contained inference: the checkpoint carries everything learned
(whitening, calibrated heads, flow weights, temperatures); this module is
the fixed machinery around it.  The trainer lives in the research folder
(mjolnir_research/mjolnir_experiments/neurosymbolic_flow/exp27_decision_serve.py)
-- this package serves, it does not train.

INFERENCE PATH (identical for every head after step 2):
    1. exact parser  (regex, 0 params, fail-closed)  -> 4 rule features
    2. frozen encoder (SmolLM2-360M, 0 trainable)    -> 960-d mean pool
    3. whitening (fit at train, frozen)              -> 128-d
    4. fusion                                        -> 132-d joint
    5. head: flow/gauss density ratio, or calibrated MLP logit / T
    6. contract: noul = 1 - P(unsafe), 4 dp (exact 0/1 as int)

Same input -> identical output: nothing retrains here, nothing samples.
"""
from __future__ import annotations

import os
import re

import numpy as np
import torch

HEADS = ("flow", "full", "text", "rules", "gauss")
DEFAULT_HEAD = os.environ.get("MJOLNIR_DECIDE_HEAD", "flow")
if DEFAULT_HEAD not in HEADS:                      # fail-closed on bad env
    DEFAULT_HEAD = "flow"

# OOD floor for the generative (flow) head — REGISTERED from measurements,
# not tuned: held-out ID min density = -409, Mercury-100 max = -6,442,
# constructed-OOD mean = -89,439 (nats, max over classes).  Below the floor
# the density ratio is meaningless (both-class collapse) and the head
# answers noul=0 — out-of-regime is not safe to auto-run.  Set
# MJOLNIR_DECIDE_OOD_FLOOR to a float to override, or "off" to disable.
from mjolnir_decide.thresholds import (  # noqa: E402
    OOD_FLOOR_CALIBRATION, OOD_FLOOR_DEFAULT, OOD_FLOOR_QUANTITY,
    declaration as ood_floor_declaration, is_possible as ood_floor_is_possible)


def _ood_floor() -> float | None:
    """Env override for the floor VALUE; the declaration (quantity + tails) is in
    `thresholds.py` and is not overridable — an undeclared quantity is the bug
    this whole mechanism exists to prevent."""
    raw = os.environ.get("MJOLNIR_DECIDE_OOD_FLOOR", str(OOD_FLOOR_DEFAULT))
    if raw.strip().lower() in ("off", "none", ""):
        return None
    try:
        return float(raw)
    except ValueError:
        return OOD_FLOOR_DEFAULT


SMOLLM = "HuggingFaceTB/SmolLM2-360M-Instruct"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))    # .../src/.. -> repo root
_FALLBACK_CKPT = os.path.join(
    _REPO, "mjolnir_research", "mjolnir_experiments", "neurosymbolic_flow",
    "cache", "exp27_serve_s0.pt")


def checkpoint_digest(path: str | None = None) -> str:
    """Pin the exact weights behind every number."""
    import hashlib
    p = resolve_checkpoint(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def resolve_checkpoint(explicit: str | None = None) -> str:
    """MJOLNIR_DECIDE_CKPT -> explicit arg -> package data/ -> repo cache."""
    for cand in (explicit,
                 os.environ.get("MJOLNIR_DECIDE_CKPT"),
                 os.path.join(_HERE, "data", "exp27_serve_s0.pt"),
                 _FALLBACK_CKPT):
        if cand and os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "no checkpoint found. Produce one with the research trainer "
        "(mjolnir_research/mjolnir_experiments/neurosymbolic_flow/"
        "exp27_decision_serve.py --train) and place it at "
        "~/.../mjolnir_decide/data/exp27_serve_s0.pt, or point "
        "MJOLNIR_DECIDE_CKPT at it.")


# ------------------------------------------------------------- parser -------
KIND_BY_TOOL = {"select_rows": "read", "update_rows": "write",
                "insert_row": "ins", "delete_rows": "del",
                "drop_table": "drop"}


def parse_feats(state: str) -> np.ndarray:
    """exact parser (0 params): rule features out of the raw state text.
    fail-closed: unknown verb -> 'write'; missing backup info -> NO backup."""
    m = re.search(r"Proposed tool call:\s*(\w+)\s*\(", state)
    kind = KIND_BY_TOOL.get(m.group(1) if m else "", "write")
    destructive = kind in ("del", "drop")
    readonly = kind == "read"
    if re.search(r"\bno backup\b", state, re.I):
        backup = 0.0
    elif re.search(r"\bbackup\b", state, re.I):
        backup = 1.0
    else:
        backup = 0.0
    return np.array([float(destructive), float(readonly), backup,
                     float(kind == "drop")], dtype=np.float32)


# --------------------------------------------------------- flow (NSPNF) -----
def _onehot(idx, K):
    return torch.nn.functional.one_hot(idx.long(), K).float()


class CondCouple(torch.nn.Module):
    """One affine coupling block (exact analytic logdet, exact inverse)."""

    def __init__(self, C, K, hidden, transform_first):
        super().__init__()
        self.half = C // 2
        self.tf = transform_first
        self.st_net = torch.nn.Sequential(
            torch.nn.Conv2d(self.half + K, hidden, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden, hidden, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(hidden, 2 * self.half, 3, padding=1),
        )
        torch.nn.init.zeros_(self.st_net[-1].weight)
        torch.nn.init.zeros_(self.st_net[-1].bias)

    def forward(self, z, c, reverse=False):
        half = self.half
        if self.tf:
            keep, moved = z[:, half:], z[:, :half]
        else:
            keep, moved = z[:, :half], z[:, half:]
        cc = c[:, :, None, None].expand(-1, -1, z.shape[2], z.shape[3])
        st = self.st_net(torch.cat([keep, cc], dim=1))
        shift, raw_ls = st[:, :half], st[:, half:]
        log_scale = torch.tanh(raw_ls)
        if not reverse:
            moved = moved * torch.exp(log_scale) + shift
            logdet = log_scale.flatten(1).sum(dim=1)
        else:
            moved = (moved - shift) * torch.exp(-log_scale)
            logdet = -log_scale.flatten(1).sum(dim=1)
        out = (torch.cat([moved, keep], dim=1) if self.tf
               else torch.cat([keep, moved], dim=1))
        return out, logdet


class CondFlow(torch.nn.Module):
    """Stack of conditional couplings.  reverse=True inverts exactly."""

    def __init__(self, C, K, n_blocks=4, hidden=128):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            CondCouple(C, K, hidden, transform_first=(i % 2 == 1))
            for i in range(n_blocks)
        )
        self.C, self.K = C, K

    def forward(self, z, p_idx, reverse=False):
        c = _onehot(p_idx, self.K)
        logdet = torch.zeros(z.shape[0], device=z.device)
        blocks = list(reversed(self.blocks)) if reverse else list(self.blocks)
        for blk in blocks:
            z, ld = blk(z, c, reverse=reverse)
            logdet = logdet + ld
        return z, logdet


class MLP(torch.nn.Module):
    def __init__(self, d, out, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, out))

    def forward(self, z):
        return self.net(z.flatten(1))


# ------------------------------------------------------------- model --------
class Model:
    """Loads the checkpoint once; answers noul questions forever after.

    head: one of HEADS.  Default is the generative flow; discriminative
    heads are the runtime alternative (--head full|text|rules or gauss).
    """

    def __init__(self, ckpt: str | None = None, head: str | None = None,
                 device: str = "cpu"):
        self.head = head or DEFAULT_HEAD
        if self.head not in HEADS:
            raise ValueError(f"unknown head {self.head!r}; one of {HEADS}")
        path = resolve_checkpoint(ckpt)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        trained = ck.get("encoder", "smollm")
        if trained != "smollm":
            raise ValueError(
                f"checkpoint {path} was trained with encoder {trained!r}; "
                f"this package serves the smollm checkpoint only")
        if ck["d_text"] != ck.get("whiten", ck["d_text"]):
            raise ValueError(f"corrupt checkpoint {path}: d_text "
                             f"{ck['d_text']} != whiten {ck.get('whiten')}")
        self.path, self.ck, self.device = path, ck, device
        self.ood_floor = _ood_floor()
        self.ckpt_path = resolve_checkpoint(None)
        self.ckpt_digest = checkpoint_digest(self.ckpt_path)   # once, not per request
        self.mu_t, self.Vw = ck["mu_t"], ck["Vw"]
        self.fm, self.fs = ck["fm"], ck["fs"]
        self.d_joint = ck["d_joint"]

        def _mlp(state, d):
            m = MLP(d, 1, hidden=64)
            m.load_state_dict(state)
            m.eval()
            return m

        self.models = {
            "full": _mlp(ck["A_full"], ck["d_text"] + ck["d_feat"]),
            "rules": _mlp(ck["A_rules"], ck["d_feat"]),
            "text": _mlp(ck["A_text"], ck["d_text"]),
        }
        self.flow = CondFlow(self.d_joint, 2, n_blocks=4, hidden=128)
        self.flow.load_state_dict(ck["flow"])
        self.flow.eval()
        self._enc = None       # lazy encoder (torch import is heavy)
        self._enc_cache = {}

    # -- encoder ------------------------------------------------------------
    def _encode(self, state: str) -> np.ndarray:
        if state in self._enc_cache:
            z = self._enc_cache[state]
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if self._enc is None:
                tok = AutoTokenizer.from_pretrained(SMOLLM)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                mdl = AutoModelForCausalLM.from_pretrained(SMOLLM).to(self.device)
                mdl.eval()
                for p in mdl.parameters():
                    p.requires_grad_(False)
                self._enc = (mdl, tok)
            mdl, tok = self._enc
            with torch.no_grad():
                b = tok([state], return_tensors="pt", padding=True,
                        truncation=True, max_length=96).to(self.device)
                h = mdl(**b, output_hidden_states=True).hidden_states[-1]
                m = b["attention_mask"].unsqueeze(-1).float()
                z = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()[
                    0].astype(np.float32)
            self._enc_cache[state] = z
        return ((torch.as_tensor(z) - self.mu_t) @ self.Vw).numpy()

    def input_tokens(self, state: str) -> int:
        if self._enc is None:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(SMOLLM)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._enc = (None, tok)
        tok = self._enc[1]
        return len(tok(state, add_special_tokens=True)["input_ids"])

    # -- heads --------------------------------------------------------------
    def _joint(self, state: str) -> np.ndarray:
        z = self._encode(state)
        e = ((parse_feats(state) - self.fm) / self.fs).astype(np.float32)
        return np.concatenate([z, e]).astype(np.float32)

    def _flow_ll(self, x: np.ndarray, c: int) -> float:
        with torch.no_grad():
            w = torch.from_numpy(x[None]).reshape(1, self.d_joint, 1, 1)
            out, ld = self.flow(w, torch.full((1,), c, dtype=torch.long))
            n = out.flatten(1).shape[1]
            return float((-0.5 * out.flatten(1).pow(2).sum(1) + ld
                          - 0.5 * n * np.log(2 * np.pi)).item())

    def _gauss_ll(self, x: np.ndarray, c: int) -> float:
        d = (torch.from_numpy(x) - self.ck["mu_g"][c]) / self.ck["sg_g"][c]
        return float((-0.5 * d.pow(2) - self.ck["sg_g"][c].log()).sum().item())

    def _flow_score(self, x: np.ndarray, c: int) -> float:
        """The quantity the OOD floor was CALIBRATED on: the flow's
        per-dimension log-density (`_flow_ll / d_joint`). Comparing the floor
        against the total joint LL fired it on every input."""
        return self._flow_ll(x, c) / self.d_joint

    def explain(self, state: str) -> dict:
        """The decision path in ONE pass — the value, the regime, the floor.

        Authoritative for `answer()` (which must not recompute the flow twice),
        and the basis of the release gate and the provenance block.
        """
        x = self._joint(state)
        if self.head != "flow":
            p = self.p_unsafe(state)
            noul = round(1.0 - p, 4)
            return {"head": self.head, "p_unsafe": p, "noul": noul,
                    "in_regime": True, "floor": self.ood_floor,
                    "quantity": None}
        prior, T = self.ck["prior"], self.ck["TF"]
        ll1, ll0 = self._flow_ll(x, 1), self._flow_ll(x, 0)
        s1, s0 = ll1 / self.d_joint, ll0 / self.d_joint
        in_regime = self.ood_floor is None or max(s0, s1) >= self.ood_floor
        if not in_regime:
            p = 1.0
        else:
            raw = np.log(prior) + ll1 - np.log(1 - prior) - ll0
            p = float(torch.sigmoid(torch.as_tensor(raw / T)))
        noul = round(1.0 - p, 4)
        if noul in (0.0, 1.0):
            noul = int(noul)
        return {"head": self.head, "quantity": OOD_FLOOR_QUANTITY,
                "flow_score": round(max(s0, s1), 2),
                "ll0": round(ll0, 1), "ll1": round(ll1, 1),
                "floor": self.ood_floor, "in_regime": in_regime,
                "p_unsafe": p, "noul": noul}

    def provenance(self) -> dict:
        """Pinned identity of everything that produced a number."""
        return {
            "checkpoint": {"name": os.path.basename(self.ckpt_path),
                           "digest": self.ckpt_digest},
            "base_model": SMOLLM, "head": self.head,
            "floor": {"quantity": OOD_FLOOR_QUANTITY, "value": self.ood_floor,
                      "calibrated_on": OOD_FLOOR_CALIBRATION},
            "replay": ("model-backed: pin the checkpoint digest; float arithmetic is "
                       "not bit-replayable across builds — the receipt pins weights "
                       "and inputs, not a proof object"),
            "blind_spot": {"prose_hazards": "unread",
                           "behaviour": "fail-closed: out-of-regime -> noul 0"},
        }

    def p_unsafe(self, state: str) -> float:
        """P(unsafe) under the configured head.  Core of every answer."""
        x = self._joint(state)
        if self.head == "flow":
            prior, T = self.ck["prior"], self.ck["TF"]
            ll1, ll0 = self._flow_ll(x, 1), self._flow_ll(x, 0)
            # the floor is compared to the quantity it was calibrated on
            if (self.ood_floor is not None
                    and max(self._flow_score(x, 0),
                            self._flow_score(x, 1)) < self.ood_floor):
                return 1.0     # out-of-regime: fail-closed, never auto-run
            raw = np.log(prior) + ll1 - np.log(1 - prior) - ll0
            return float(torch.sigmoid(torch.as_tensor(raw / T)))
        if self.head == "gauss":
            prior, T = self.ck["prior"], self.ck["TG"]
            raw = (np.log(prior) + self._gauss_ll(x, 1)
                   - np.log(1 - prior) - self._gauss_ll(x, 0))
            return float(torch.sigmoid(torch.as_tensor(raw / T)))
        # discriminative: (full) joint, (text) whitened only, (rules) feats
        xx = {"full": x,
              "text": x[: self.ck["d_text"]],
              "rules": ((parse_feats(state) - self.fm)
                        / self.fs).astype(np.float32)}[self.head]
        with torch.no_grad():
            raw = float(self.models[self.head](
                torch.from_numpy(xx[None])).flatten().item())
        return float(torch.sigmoid(torch.as_tensor(raw / self.ck["TA"][self.head])))

    def noul(self, state: str) -> float | int:
        """1 - P(unsafe), rounded to 4 dp (exact 0/1 as int) — the contract."""
        noul = round(1.0 - self.p_unsafe(state), 4)
        if noul in (0.0, 1.0):
            return int(noul)
        return noul
