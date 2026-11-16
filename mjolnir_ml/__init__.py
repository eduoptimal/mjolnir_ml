"""mjolnir_ml — fail-closed decision models and independent audit replay.

The top-level package is lazy: importing `mjolnir_ml` does not import the model,
sklearn, numpy, or the auditor. Developer and auditor paths load only what they
use.
"""
from __future__ import annotations

import importlib

__version__ = "0.1.0"

_WIRE = {
    "RELEASE": "mjolnir_ml.core.wire",
    "ASK": "mjolnir_ml.core.wire",
    "DENY": "mjolnir_ml.core.wire",
    "ABSTAIN": "mjolnir_ml.core.wire",
    "WIRE": "mjolnir_ml.core.wire",
    "RANK": "mjolnir_ml.core.wire",
    "project": "mjolnir_ml.core.wire",
    "strictest": "mjolnir_ml.core.wire",
    "WarrantLedger": "mjolnir_ml.warrant.ledger",
}

_MODULES = (
    "mjolnir_ml.release.abstention",
    "mjolnir_ml.operations.causal",
    "mjolnir_ml.calibration.calibration",
    "mjolnir_ml.warrant.certificate",
    "mjolnir_ml.release.costs",
    "mjolnir_ml.release.feasibility",
    "mjolnir_ml.calibration.regime",
    "mjolnir_ml.calibration.risk",
    "mjolnir_ml.core.dsl",
    "mjolnir_ml.core.model",
    "mjolnir_ml.release.hazards",
    "mjolnir_ml.operations.discovery",
    "mjolnir_ml.operations.live",
    "mjolnir_ml.release.readiness",
    "mjolnir_ml.release.pair",
    "mjolnir_ml.core.policy",
    "mjolnir_ml.release.report",
    "mjolnir_ml.warrant.dev_replay",
)

_ALIASES = {"DENY": "DENY_W", "ABSTAIN": "ABSTAIN_W"}

_COLLISIONS = {
    "mjolnir_ml.core.dsl": "dsl",
    "mjolnir_ml.release.readiness": "readiness",
    "mjolnir_ml.release.report": "report",
    "mjolnir_ml.release.feasibility": "feasibility",
}


def __getattr__(name):
    if name in _ALIASES:
        module = importlib.import_module("mjolnir_ml.core.wire")
        value = getattr(module, _ALIASES[name])
        globals()[name] = value
        return value
    if name in _WIRE:
        module = importlib.import_module(_WIRE[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    for module_name in _MODULES:
        module = importlib.import_module(module_name)
        if module_name in _COLLISIONS:
            attr = _COLLISIONS[module_name]
            globals()[attr] = getattr(module, attr)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module 'mjolnir_ml' has no attribute {name!r}")


__all__ = [
    "dsl", "rule", "compile_class", "DecisionPolicyBase", "DslError",
    "DecisionModel", "Disposition", "BindError",
    "Policy", "Statement", "MODEL_FACTS",
    "Kappa", "AdaptiveKappa", "fit_kappa", "CalibrationError",
    "REGRESSOR_METHODS",
    "Costs", "cost_policy",
    "RegimeError", "MultivariateRegimeFloor",
    "RiskError", "RiskIndex", "adverse_event", "crc_threshold",
    "crc_thresholds", "fit_risk_index",
    "FeasibilityError", "FeasibilityReport", "RuleCheck", "feasibility",
    "certify", "verify_certificate", "CertificateError", "CERT_SCHEMA",
    "Hazard", "HazardLedger", "measure_completeness",
    "readiness", "ReadinessReport", "ReadinessThresholds", "Gate",
    "ReadinessError",
    "LiveSession", "Journal", "DetectorSpec", "Detection",
    "default_detectors", "Containment", "Retraction",
    "ChangeSet", "ChangeGate", "Evaluation", "Promotion",
    "Approval", "ApprovalAuthority",
    "DiscoverySession", "Census", "Candidate", "CENSUS_VERDICTS",
    "CERTIFIED", "DECLARED", "DECLINED", "REFUTED",
    "DiscoveryGate", "DiscoveryEvaluation", "Admission",
    "KnowledgeLedger", "KnowledgeRecord", "DiscoveryError",
    "DiscoveryDetector", "DiscoveryDetection",
    "default_discovery_detectors", "DiscoveryContainment",
    "certify_admission", "verify_admission_certificate",
    "CausalSession", "CausalClaim", "CausalOracle", "CausalError",
    "Strategy", "do", "Table", "Series", "Estimate",
    "event_study", "series_from_journal",
    "Pair", "omega_pair", "wilson", "audit", "audit_model",
    "RiskControlledReleaseGate", "ReleaseDecision", "AbstentionCertificate",
    "AbstentionError", "ABSTENTION_SCHEMA",
    "report", "report_data",
    "degenerate_controls",
    "RELEASE", "ASK", "DENY", "ABSTAIN", "WIRE", "RANK", "project",
    "strictest", "WarrantLedger", "__version__",
]
