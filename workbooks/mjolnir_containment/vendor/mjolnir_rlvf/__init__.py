"""mjolnir_rlvf — Reinforcement Learning from Verifiable Feedback, product.

Distilled from mjolnir_research/mjolnir_experiments/rlvf{,_agent}: the same
proven pipeline (probe -> rollout -> pairs -> DPO -> eval), with two product
boundaries preserved:

  * the mjolnir authority is imported READ-ONLY (every reward and outcome is
    a typed verdict from mjolnir_prover.lib.certify — this package never
    decides anything itself);
  * training dependencies are optional (extra `rlvf`); rollout/reward/eval
    need only the core install.

Entry point:  mjolnir rlvf --model=<id> --params='{...}' [--stage ...]
"""
