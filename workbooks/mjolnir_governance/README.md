# mjolnir_governance

Volume II workbook for *Non-Stationary Machine Learning: Temporal Containment
and the Limits of Exchangeability*. This directory contains the ten lab
drivers named in the book. It is a lab distribution over the engine, not a
package: the reader imports only `mjolnir_ml`.

## Layout

```
workbooks/mjolnir_governance/
  labs/           ten lab drivers (lab01 ... lab10)
  tests/          workbook smoke tests
  run_all.py      runs every lab and summarizes exit codes
  README.md
```

## Run a lab

```bash
cd workbooks/mjolnir_governance
python labs/lab01_drift.py --replay
python labs/lab10_causal.py --replay
```

Every lab accepts `--replay`, prints deterministic JSON, and exits `0` on the
expected result. The harness puts the repository root on `sys.path`, so the
engine is importable from a clean checkout even without installation.

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
| 1 | `labs/lab01_drift.py` | Induce the drift flip |
| 2 | `labs/lab02_pair.py` | The degenerate controls |
| 3 | `labs/lab03_bench.py` | Pre-register the head-to-head |
| 4 | `labs/lab04_completeness.py` | The observability ceiling |
| 5 | `labs/lab05_certificate.py` | Three tampering cases |
| 6 | `labs/lab06_readiness.py` | The blocking gate |
| 7 | `labs/lab07_live.py` | The injected hazard |
| 8 | `labs/lab08_flywheel.py` | The audit memo |
| 9 | `labs/lab09_discovery.py` | The refused admission |
| 10 | `labs/lab10_causal.py` | The three causal cases |

## Engine provenance

The labs import the pinned `mjolnir_ml` engine from this repository. The
workbook adds fixtures and drivers only; the engine's own `tests/` remain the
source of truth for the laws the labs exercise.
