"""mjolnir_prover.cli.commands — Typer command modules.

Each module exposes ``register(app)`` which attaches its commands to the root
Typer application. Commands stay thin: they translate CLI arguments into engine
calls and hand the structured result to ``mjolnir_prover.cli.reporting``.
"""