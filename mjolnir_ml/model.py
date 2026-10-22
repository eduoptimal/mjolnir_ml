"""mjolnir_ml.model — DecisionModel: learner + policy + κ -> fail-closed disposer.

THE TRANSFORM this package exists for: a predictor answers "what will happen";
a decision model answers "may this run". The weights never change; the
harness does. Pipeline per row, in order, each step failing CLOSED into a
named abstention (never into a release, never into a swallow):

    1. contract   row is a mapping; every estimator feature stated; finite
    2. regime     value within the recorded training bounds (fit_regime)
    3. predict    estimator output -> model facts (pred, conf, p_positive)
    4. κ          calibrated trust (calibrate): certified / set_size for
                  classifiers, lo / hi for regressors. α is the declared risk.
    5. premises   engine: violated premise -> abstain(premise_code)
    6. ask/deny   engine: severity max (deny beats ask)
    7. release    ONLY if engine said ROUTE and a release statement fires;
                  ROUTE with no firing release -> abstain(uncovered)

Laws (pinned by tests/test_model.py):
  * fail closed     — uncovered, missing, non-finite, out-of-regime, model
                      exceptions are all abstain, never release.
  * model proposes  — the estimator emits FACTS; verdicts come only from the
                      policy. Swapping the estimator cannot produce a verdict
                      the policy does not contain.
  * only-tighten    — a release requires a firing release statement on top of
                      a clean engine ROUTE; ask/deny/premises can only make
                      the outcome stricter.
  * unknown ≠ zero  — an uncalibrated group yields no certified set/interval;
                      a required-but-absent fact is abstain.
  * reproducible    — disposition carries claim/policy/model/calibration
                      digests and a certificate id; verify() recomputes.

MODEL FACTS (reserved names):
    pred         predicted label (classifier) / value (regressor)
    conf         max class probability (classifiers with predict_proba)
    p_positive   binary classifiers only: P(classes_[-1])
    certified    conformal prediction set is a singleton at the declared α
    set_size     conformal prediction-set size
    lo / hi      conformal interval bounds (regressor, calibrated group)

Unavailable-for-this-estimator facts fail at BIND. Available-but-not-yet-
calibrated facts fail at DISPOSE as abstain(missing:<fact>) — the floor is
structural, not a constructor convention.

GUARANTEE TAG (differentiated from the conformal `certified` FACT):
    declared      the deciding statements read only stated input facts
    model_backed  a deciding statement read estimator output (pred/conf/
                  p_positive/certified/set_size/lo/hi)
Harness abstentions (missing/invalid/regime/error/uncovered) are certified
by construction: the floor never depends on a model fact.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from mjolnir_ml.policy import MODEL_FACTS, Policy
from mjolnir_ml.vendors import wire
from mjolnir_ml.vendors.core.verdict import UNSTATED, Verdict, evaluate

_MAX_MSG = 120


class BindError(ValueError):
    """The estimator and policy cannot be bound (refuse to construct)."""


# ----------------------------------------------------------------- record ----
@dataclass(frozen=True)
class Disposition:
    """One disposed row. Digest-pinned: the table and the calibration that
    ruled travel with it; `certificate_id` addresses the whole proof term."""

    schema: str
    verdict: str                      # release | ask | deny | abstain
    basis: tuple                      # deciding statement codes / harness tags
    missing: str | None
    reason: str
    say: str                          # human sentence, values included
    guarantee: str                    # declared | model_backed
    claim_digest: str
    policy_digest: str
    model_digest: str
    calibration_digest: str | None = None
    alpha: float | None = None
    certificate_id: str = ""
    prediction: dict | None = None    # model facts when predict ran
    facts: dict = field(default_factory=dict)   # complete fact table (audit)
    latency_ms: float = 0.0
    context: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "schema": self.schema, "verdict": self.verdict,
            "basis": list(self.basis), "missing": self.missing,
            "reason": self.reason, "say": self.say,
            "guarantee": self.guarantee, "claim_digest": self.claim_digest,
            "policy_digest": self.policy_digest,
            "model_digest": self.model_digest,
            "calibration_digest": self.calibration_digest,
            "alpha": self.alpha, "certificate_id": self.certificate_id,
            "facts": dict(self.facts), "prediction": self.prediction,
            "latency_ms": self.latency_ms,
            "context": dict(self.context),
        }

    @property
    def claims(self) -> bool:
        """A committed operational outcome (the Ω-pair's claim axis)."""
        return self.verdict in (wire.RELEASE, wire.DENY_W)

    def retag(self, verdict: str, basis, *, reason=None, say=None):
        """A NEW disposition with a recomputed certificate id.

        Used by containment, which may only tighten. The original record is
        untouched; every retag is journaled by the caller."""
        from dataclasses import replace
        basis = tuple(basis)
        cert_id = _digest_parts(self.claim_digest, self.policy_digest,
                                self.model_digest, self.calibration_digest or "-",
                                verdict, ",".join(basis), self.guarantee)
        return replace(self, verdict=verdict, basis=basis,
                       reason=reason if reason is not None else self.reason,
                       say=say if say is not None else self.say,
                       certificate_id=cert_id)


# ------------------------------------------------------------------ model ----
def _scalar(v):
    """numpy scalars -> python; anything else passes through."""
    return v.item() if hasattr(v, "item") else v


def _float(v) -> float:
    return float(_scalar(v))


def _digest_parts(*parts) -> str:
    blob = "|".join(p if isinstance(p, str) else repr(p) for p in parts)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8",
                                                  "surrogateescape")).hexdigest()


def _model_digest(estimator, features: tuple) -> str:
    """Config + fitted-state digest. Deterministic reprs (sklearn params)."""
    parts = [type(estimator).__module__ + "." + type(estimator).__name__]
    if hasattr(estimator, "get_params"):
        try:
            params = estimator.get_params(deep=False)
            parts.append(json.dumps({k: repr(v) for k, v in sorted(params.items())},
                                    sort_keys=True))
        except Exception:                                   # noqa: BLE001
            parts.append(repr(estimator))
    else:
        parts.append(repr(estimator))
    parts.append(repr(tuple(features)))
    for attr in ("classes_", "coef_", "intercept_", "n_features_in_",
                 "feature_importances_"):
        val = getattr(estimator, attr, None)
        if val is not None:
            parts.append(f"{attr}={val!r}")
    return _digest_parts(*parts)


def _claim_digest(row) -> str:
    blob = json.dumps(dict(row), sort_keys=True, separators=(",", ":"),
                      default=repr, ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


class DecisionModel:
    """Wrap a fitted estimator + a compiled policy into a fail-closed
    decision model. Bind validates; dispose only ever abstains.

    Disposition.guarantee: declared (stated facts only) or model_backed
    (a deciding statement read estimator output). The conformal proof
    mark is a separate fact, `certified`, and never shares that name."""

    SCHEMA = "mjolnir.ml.disposition/1"

    def __init__(self, estimator, policy, features=None, regime="required"):
        """policy: a @dsl class or its .policy(). features: input contract
        (default estimator.feature_names_in_). regime: "required" (default —
        dispose abstains until fit_regime ran) or "off" (disclosed)."""
        if isinstance(policy, type):
            policy = policy.policy()
        if not isinstance(policy, Policy):
            raise BindError("policy must be a @dsl class or a Policy")
        self.estimator = estimator
        self.policy = policy
        if regime not in ("required", "off"):
            raise BindError(f"regime must be 'required' or 'off', got {regime!r}")
        self.regime = regime

        feats = features
        if feats is None:
            feats = getattr(estimator, "feature_names_in_", None)
            if feats is not None:
                feats = [str(f) for f in feats]
        if not feats:
            raise BindError(
                "cannot infer input features — pass features=(...) "
                "(estimator has no feature_names_in_)")
        self.features = tuple(feats)
        if len(set(self.features)) != len(self.features):
            raise BindError(f"duplicate features: {self.features}")
        clash = set(self.features) & MODEL_FACTS
        if clash:
            raise BindError(
                f"input features collide with reserved model facts "
                f"{sorted(clash)} — rename the columns")

        has_proba = callable(getattr(estimator, "predict_proba", None))
        is_clf = getattr(estimator, "classes_", None) is not None
        if has_proba and not is_clf:
            raise BindError(
                "estimator has predict_proba but no classes_ — fit it before "
                "binding (an unfitted classifier cannot be bound)")
        self._is_clf = is_clf
        self._proba = has_proba
        self._kappa = None

        _classes = getattr(estimator, "classes_", None)
        n_classes = 0 if _classes is None else len(_classes)
        capabilities = {"pred"}
        if is_clf and has_proba:
            capabilities |= {"conf", "certified", "set_size"}
            if n_classes == 2:
                capabilities.add("p_positive")   # P(classes_[-1])
        elif is_clf:
            pass                                  # labels only
        else:
            capabilities |= {"lo", "hi"}

        available = set(self.features) | capabilities
        for s in policy.statements:
            unknown = set(s.requires) - available
            if unknown:
                hint = ""
                if unknown & {"conf", "p_positive", "certified",
                              "set_size"} and not has_proba:
                    hint = " (classifier without predict_proba: no conformal set)"
                elif unknown & {"lo", "hi"} and is_clf:
                    hint = " (lo/hi are regressor interval facts)"
                elif unknown & {"p_positive"} and is_clf:
                    hint = " (p_positive is a binary-classifier fact)"
                elif unknown & {"certified", "set_size"} and not is_clf:
                    hint = " (certified/set_size are classifier set facts)"
                raise BindError(
                    f"policy statement {s.code!r} requires "
                    f"{sorted(unknown)}; available facts for this estimator: "
                    f"{sorted(available)}{hint}")

        self._engine_rules = policy.engine_rules()
        self._model_digest = _model_digest(estimator, self.features)
        self._bounds: dict | None = None

    # ---------------------------------------------------------- calibration --
    def calibrate(self, rows, y, alpha: float = 0.05,
                  group: str | None = None, adaptive: bool = False,
                  gamma: float = 0.05, window: int = 400) -> "DecisionModel":
        """Fit the conformal trust index κ on held-out rows. `group` is a
        feature name for Mondrian (per-region) calibration; a group with too
        little data is a CalibrationError, never an uncalibrated claim.

        adaptive=True fits ACI (beta_release G4): the per-group risk level
        alpha_t moves from observed miscoverage, so `observe(row, y)` after a
        decision restores coverage under drift. gamma is the step size;
        window bounds the score memory."""
        from mjolnir_ml.calibration import fit_kappa as _fit_kappa
        if group is not None and group not in self.features:
            raise BindError(
                f"calibration group {group!r} is not an input feature "
                f"{list(self.features)}")
        self._kappa = _fit_kappa(self.estimator, rows, y, alpha=alpha,
                                group=group,
                           features=self.features, adaptive=adaptive,
                           gamma=gamma, window=window)
        return self

    def observe(self, row, y_true) -> "DecisionModel":
        """Stream a realized label into κ (ACI). No-op for static calibration,
        so a caller can wire feedback unconditionally. Unknown group -> the
        engine refuses to update that region (fail closed)."""
        if self._kappa is None:
            raise BindError("observe() needs a calibration first")
        if not hasattr(self._kappa, "observe"):
            return self          # static calibration: feedback is a no-op
        if isinstance(row, Mapping):
            key = self._kappa.group_key(row)
        else:
            key = "__all__"
        X = [[row[f] for f in self.features]]
        pred = _scalar(self.estimator.predict(X)[0])
        if self._is_clf:
            label = _scalar(y_true)
            classes = [_scalar(c) for c in self.estimator.classes_]
            idx = classes.index(label)
            proba = [_float(p) for p in self.estimator.predict_proba(X)[0]]
            score = 1.0 - proba[idx]
        else:
            score = abs(float(y_true) - float(pred))
        self._kappa.observe(key, score)
        return self

    @property
    def kappa(self):
        return self._kappa

    @property
    def model_digest(self) -> str:
        return self._model_digest

    # -------------------------------------------------------------- regime --
    def fit_regime(self, rows) -> "DecisionModel":
        """Record per-feature training bounds from list[dict] (same shape as
        dispose). Numeric: [min, max] over finite values. Non-numeric: the
        set of seen values; an unseen category is out-of-regime."""
        if not isinstance(rows, (list, tuple)) or not rows:
            raise BindError("fit_regime needs a non-empty list of dict rows")
        bounds: dict = {}
        for f in self.features:
            vals = [r[f] for r in rows if isinstance(r, Mapping) and f in r]
            if not vals:
                raise BindError(f"fit_regime: feature {f!r} absent from rows")
            numeric = all(isinstance(v, numbers.Real) and not isinstance(v, bool)
                          for v in vals)
            if numeric:
                fin = [float(v) for v in vals if math.isfinite(float(v))]
                if not fin:
                    raise BindError(f"fit_regime: feature {f!r} has no "
                                    f"finite values")
                bounds[f] = ("num", min(fin), max(fin))
            else:
                bounds[f] = ("seen", frozenset(vals))
        self._bounds = bounds
        return self

    # ------------------------------------------------------------- dispose --
    def dispose(self, row, context=None) -> Disposition:
        t0 = time.perf_counter()
        if not isinstance(row, Mapping):
            raise TypeError(f"row must be a mapping, got {type(row).__name__}")
        self._claim = _claim_digest(row)

        def _abs(reason, say, basis, missing=None, prediction=None,
                 guarantee="declared"):
            return self._pack(wire.ABSTAIN_W, basis, missing, reason, say,
                              guarantee, prediction, t0, context)

        # 1. contract: every estimator feature stated, finite
        for f in self.features:
            if f not in row:
                return _abs(f"missing:{f}",
                            f"cannot decide: required fact {f!r} is not stated",
                            (f"missing:{f}",), missing=f)
            v = row[f]
            if v is None:
                return _abs(f"invalid:{f}",
                            f"cannot decide: fact {f!r} is void (None)",
                            (f"invalid:{f}",), missing=f)
            if isinstance(v, numbers.Real) and not isinstance(v, bool) \
                    and not math.isfinite(float(v)):
                return _abs(f"invalid:{f}",
                            f"cannot decide: fact {f!r} is not finite "
                            f"({v!r})", (f"invalid:{f}",))

        # 2. regime (fail-closed floor: required and not fitted is itself
        #    an abstention — absent is not a permission)
        if self.regime == "required":
            if self._bounds is None:
                return _abs("regime_not_fitted",
                            "cannot decide: no training regime recorded "
                            "(call fit_regime)", ("regime_not_fitted",))
            for f, b in self._bounds.items():
                v = row[f]
                if b[0] == "num":
                    if isinstance(v, bool) or not isinstance(v, numbers.Real):
                        return _abs(f"out_of_regime:{f}",
                                    f"cannot decide: {f}={v!r} is not numeric "
                                    f"within recorded bounds "
                                    f"[{b[1]:g}, {b[2]:g}]",
                                    (f"out_of_regime:{f}",))
                    if not (b[1] <= float(v) <= b[2]):
                        return _abs(f"out_of_regime:{f}",
                                    f"cannot decide: {f}={float(v):g} is "
                                    f"outside recorded training bounds "
                                    f"[{b[1]:g}, {b[2]:g}]",
                                    (f"out_of_regime:{f}",))
                elif v not in b[1]:
                    return _abs(f"out_of_regime:{f}",
                                f"cannot decide: {f}={v!r} is an unseen "
                                f"category", (f"out_of_regime:{f}",))

        # 3. predict -> model facts. Any exception: abstain, never swallow.
        facts = {f: row[f] for f in self.features}
        prediction = None
        try:
            X = [[row[f] for f in self.features]]
            pred = _scalar(self.estimator.predict(X)[0])
            prediction = {"pred": pred}
            if pred is None:
                return _abs("invalid_prediction",
                            "cannot decide: estimator emitted no prediction",
                            ("invalid_prediction",))
            facts["pred"] = pred
            if self._is_clf:
                classes = [_scalar(c) for c in self.estimator.classes_]
                if self._proba:
                    proba = [_float(p) for p in
                             self.estimator.predict_proba(X)[0]]
                    conf = max(proba)
                    if not math.isfinite(conf):
                        return _abs(
                            "invalid_prediction",
                            f"cannot decide: estimator emitted non-finite "
                            f"probabilities (conf={conf!r})",
                            ("invalid_prediction",), prediction=prediction)
                    prediction.update(conf=conf)
                    facts.update(conf=conf)
                    if len(classes) == 2:
                        p_positive = proba[-1]
                        if not math.isfinite(p_positive):
                            return _abs(
                                "invalid_prediction",
                                f"cannot decide: non-finite p_positive "
                                f"({p_positive!r})",
                                ("invalid_prediction",),
                                prediction=prediction)
                        prediction["p_positive"] = p_positive
                        facts["p_positive"] = p_positive
                    if self._kappa is not None:
                        key = self._kappa.group_key(row)
                        pvec = {c: proba[j] for j, c in enumerate(classes)}
                        pset = self._kappa.prediction_set(pvec, key)
                        pset = [_scalar(c) for c in pset]
                        facts["certified"] = len(pset) == 1
                        facts["set_size"] = len(pset)
                        prediction.update(
                            set=pset, set_size=len(pset),
                            certified=len(pset) == 1,
                            calibrated_group=key)
            else:
                if self._kappa is not None:
                    key = self._kappa.group_key(row)
                    iv = self._kappa.interval(float(pred), key)
                    if iv is not None:
                        lo, hi = iv
                        facts.update(lo=lo, hi=hi)
                        prediction.update(lo=lo, hi=hi, calibrated_group=key)
        except Exception as ex:                               # noqa: BLE001
            msg = str(ex)[:_MAX_MSG]
            return _abs("model_error",
                        f"cannot decide: estimator failed ({type(ex).__name__}: "
                        f"{msg})", ("model_error",))

        # 4+5+6. the policy is the only disposer; the same function replays
        #        a certificate offline (Policy.dispose_facts).
        verdict, basis, missing = self.policy.dispose_facts(facts)
        codes = tuple(basis)

        if verdict == wire.ABSTAIN_W:
            if missing:
                say = (f"cannot decide: missing {missing!r} "
                       f"(rule needs it stated)")
                reason = f"missing:{missing}"
            elif codes == ("uncovered",):
                say = ("cannot decide: no policy statement fires for this "
                       "prediction (uncovered is abstain, never release)")
                reason = "uncovered"
            else:
                st = next((self.policy.statement(c) for c in codes
                           if c in _codes(self.policy)), None)
                if st is not None:
                    vals = ", ".join(f"{k}={facts.get(k)!r}"
                                     for k in st.requires)
                    say = (f"cannot decide: premise {st.code} does not hold "
                           f"({vals})")
                else:
                    say = "cannot decide: " + (", ".join(codes) or "abstained")
                reason = ", ".join(codes) or "abstained"
            guarantee = ("model_backed" if self.policy.uses_model_facts(
                [c for c in codes if c in _codes(self.policy)]) else "declared")
            return self._pack(verdict, codes, missing, reason, say, guarantee,
                              prediction, t0, context, facts)

        vals = ", ".join(f"{k}={facts.get(k)!r}"
                         for c in codes if c in _codes(self.policy)
                         for k in self.policy.statement(c).requires)
        if verdict == wire.RELEASE:
            say = f"released by {', '.join(codes)} ({vals})"
        elif verdict == wire.ASK:
            say = f"escalated by {', '.join(codes)} ({vals})"
        else:
            say = f"denied by {', '.join(codes)} ({vals})"
        guarantee = ("model_backed" if self.policy.uses_model_facts(
            [c for c in codes if c in _codes(self.policy)]) else "declared")
        return self._pack(verdict, codes, missing, ", ".join(codes), say,
                          guarantee, prediction, t0, context, facts)

    # ------------------------------------------------------------ versions --
    def with_policy(self, policy) -> "DecisionModel":
        """A NEW model with the same estimator, features, regime, bounds and
        kappa, and a different policy. The original is untouched."""
        if isinstance(policy, type):
            policy = policy.policy()
        if not isinstance(policy, Policy):
            raise BindError("with_policy takes a @dsl class or a Policy")
        new = DecisionModel(self.estimator, policy, features=self.features,
                            regime=self.regime)
        new._bounds = self._bounds
        new._kappa = self._kappa
        return new

    def with_estimator(self, estimator, *, calibration_rows=None,
                       calibration_y=None, alpha=None) -> "DecisionModel":
        """A NEW model with a different estimator. Model candidates must
        re-calibrate: kappa is a property of the model's error, not of the
        class of models. Without calibration data the candidate is uncalibrated
        and any policy requiring conformal facts will abstain."""
        new = DecisionModel(estimator, self.policy, features=self.features,
                            regime=self.regime)
        new._bounds = self._bounds
        if calibration_rows is not None or calibration_y is not None:
            if calibration_rows is None or calibration_y is None:
                raise BindError("with_estimator needs both calibration_rows "
                                "and calibration_y")
            new.calibrate(calibration_rows, calibration_y,
                          alpha=alpha if alpha is not None else 0.10)
        return new

    # -------------------------------------------------------------- helpers --
    def _pack(self, verdict, basis, missing, reason, say, guarantee,
              prediction, t0, context, facts=None) -> Disposition:
        cal_digest = self._kappa.digest if self._kappa is not None else None
        alpha = self._kappa.alpha if self._kappa is not None else None
        cert_id = _digest_parts(self._claim, self.policy.digest,
                                self._model_digest, cal_digest or "-",
                                verdict, ",".join(basis), guarantee)
        return Disposition(
            schema=self.SCHEMA, verdict=verdict, basis=tuple(basis),
            missing=missing, reason=reason, say=say, guarantee=guarantee,
            claim_digest=self._claim, policy_digest=self.policy.digest,
            model_digest=self._model_digest, calibration_digest=cal_digest,
            alpha=alpha, certificate_id=cert_id, prediction=prediction,
            facts=dict(facts or {}),
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
            context=dict(context or {}))

    def replay(self, disp: Disposition, row) -> bool:
        """Certificate replay: recompute and compare the pinned fields."""
        again = self.dispose(row)
        return (again.verdict == disp.verdict
                and again.basis == disp.basis
                and again.missing == disp.missing
                and again.claim_digest == disp.claim_digest
                and again.policy_digest == disp.policy_digest
                and again.model_digest == disp.model_digest
                and again.calibration_digest == disp.calibration_digest
                and again.alpha == disp.alpha
                and again.certificate_id == disp.certificate_id
                and again.guarantee == disp.guarantee)

    def __call__(self, row, context=None) -> Disposition:
        return self.dispose(row, context=context)


def _codes(policy: Policy) -> frozenset:
    return frozenset(s.code for s in policy.statements)


__all__ = ["DecisionModel", "Disposition", "BindError"]
