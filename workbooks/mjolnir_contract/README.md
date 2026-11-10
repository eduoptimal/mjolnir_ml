# mjolnir_contract

Volume I workbook for *Fail-Closed Machine Learning: Bounded Agency and the
Limits of Validity*. This directory contains the ten lab drivers named in the
book. It is a lab distribution over the engine, not a package: the reader
imports only `mjolnir_ml`.

## Layout

```
workbooks/mjolnir_contract/
  labs/           ten lab drivers (lab01 ... lab10)
  tests/          workbook smoke tests
  run_all.py      runs every lab and summarizes exit codes
  README.md
```

## Run a lab

```bash
cd workbooks/mjolnir_contract
python labs/lab01_totality.py --replay
python labs/lab10_capstone.py --replay
```

Every lab accepts `--replay`, prints deterministic JSON, and exits `0` on the
expected result. The harness puts the repository root on `sys.path`, so the
engine is importable from a clean checkout even without installation. To use an
installed engine:

```bash
pip install -e .
# or from the repository root
pip install -e .
```

Run the set:

```bash
python run_all.py
```

Run the workbook tests:

```bash
python -m pytest tests -q
```

## Lab map

| Lab | Driver | Chapter |
|---|---|---|
| 1 | `labs/lab01_totality.py` | Inducing and measuring totality failure |
| 2 | `labs/lab02_verdicts.py` | Four verdicts and the review queue |
| 3 | `labs/lab03_regime.py` | Feature-level support violations |
| 4 | `labs/lab04_facts.py` | The uncalibrated contract |
| 5 | `labs/lab05_compile.py` | Compile, digest, break |
| 6 | `labs/lab06_coverage.py` | Coverage, set size, zero-claim cell |
| 7 | `labs/lab07_groups.py` | Mondrian calibration and continuous envelopes |
| 8 | `labs/lab08_risk.py` | Two tracks, one cohort |
| 9 | `labs/lab09_costs.py` | Price the reject option |
| 10 | `labs/lab10_capstone.py` | Deploying and defending a decision contract |

## Engine provenance

The labs import the pinned `mjolnir_ml` engine from this repository. They do not
vendor it: Volume I *is* the engine surface. The workbook adds fixtures and
drivers only, so the engine's own `tests/` remain the source of truth for the
laws the labs exercise.
