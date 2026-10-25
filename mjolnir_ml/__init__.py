"""mjolnir_ml — turn a fitted predictor into a fail-closed decision model.

    from mjolnir_ml import dsl, rule, DecisionModel, DecisionPolicyBase, \
        RELEASE, ASK, DENY, omega_pair

    @dsl
    class Policy(DecisionPolicyBase):
        assert lambda amount: amount > 0, "amount_positive"
        assert rule("auto_release",
                    when=lambda p_positive, conf: p_positive < 0.05
                                 and conf >= 0.95,
                    verdict=RELEASE)

    dm = DecisionModel(fitted_sklearn_estimator, Policy, features=(...))
    dm.fit_regime(train_rows)                # fail-closed regime floor
    disp = dm.dispose(row)                   # release | ask | deny | abstain

STANDALONE: imports nothing outside this package at runtime; the frozen rule
engine and the wire projection are vendored under mjolnir_ml/vendors.
"""
from mjolnir_ml.causal import (CausalClaim, CausalError, CausalOracle,
                                CausalSession, Estimate, Series, Strategy,
                                Table, do, event_study, series_from_journal)
from mjolnir_ml.calibration import (AdaptiveKappa, CalibrationError, Kappa,
                                    REGRESSOR_METHODS, fit_kappa)
from mjolnir_ml.certificate import (CERT_SCHEMA, CertificateError, certify,
                                    verify_certificate)
from mjolnir_ml.costs import Costs, cost_policy
from mjolnir_ml.feasibility import (FeasibilityError, FeasibilityReport,
                                    RuleCheck, feasibility)
from mjolnir_ml.regime import MultivariateRegimeFloor, RegimeError
from mjolnir_ml.risk import (RiskError, RiskIndex, adverse_event,
                             crc_threshold, crc_thresholds, fit_risk_index)
from mjolnir_ml.dsl import (DecisionPolicyBase, DslError, compile_class, dsl,
                            rule)
from mjolnir_ml.model import (BindError, DecisionModel, Disposition)
from mjolnir_ml.hazards import (Hazard, HazardLedger,
                                 measure_completeness)
from mjolnir_ml.discovery import (CERTIFIED, CENSUS_VERDICTS, DECLARED,
                                  DECLINED, REFUTED, Admission, Candidate,
                                  Census, Containment as DiscoveryContainment,
                                  DiscoveryDetection, DiscoveryDetector,
                                  DiscoveryError, DiscoveryEvaluation,
                                  DiscoveryGate, DiscoverySession,
                                  KnowledgeLedger, KnowledgeRecord,
                                  certify_admission,
                                  default_discovery_detectors,
                                  verify_admission_certificate)
from mjolnir_ml.live import (Approval, ApprovalAuthority, ChangeGate,
                             ChangeSet, Containment, Detection, DetectorSpec,
                             Evaluation, Journal, LiveSession, Promotion,
                             Retraction, default_detectors)
from mjolnir_ml.readiness import (Gate, ReadinessError, ReadinessReport,
                                  ReadinessThresholds, readiness)
from mjolnir_ml.pair import (Pair, audit, audit_model, degenerate_controls,
                             omega_pair, wilson)
from mjolnir_ml.policy import MODEL_FACTS, Policy, Statement
from mjolnir_ml.report import report, report_data
from mjolnir_ml.vendors.wire import (ABSTAIN_W, ASK, DENY_W, RANK, RELEASE,
                                     WIRE, project, strictest)

ASK = ASK            # wire vocabulary: release | ask | deny | abstain
DENY = DENY_W
ABSTAIN = ABSTAIN_W

__version__ = "0.1.0"

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
    "report", "report_data",
    "degenerate_controls",
    "RELEASE", "ASK", "DENY", "ABSTAIN", "WIRE", "RANK", "project",
    "strictest", "__version__",
]
