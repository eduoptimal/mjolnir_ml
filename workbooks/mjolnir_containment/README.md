# mjolnir_containment

Volume III workbook for *Agentic Machine Learning: Autonomous Containment and
the Limits of Decidability*. This directory contains the twelve lab drivers
named in the book and a vendored copy of the containment libraries they
document. It is additive: `mjolnir_ml` itself, its tests, and its Volume I
surface are unchanged.

## Layout

```
workbooks/mjolnir_containment/
  labs/           twelve lab drivers (lab01 ... lab12)
  vendor/         containment libraries copied from the research repo
  fixtures/       small deterministic fixtures
  tests/          workbook smoke tests
  run_all.py      runs every lab and summarizes the expected exit codes
  README.md
```

## Run a lab

```bash
cd workbooks/mjolnir_containment
python labs/lab01_paralysis.py --replay
python labs/lab12_ledger.py --replay
```

Every lab accepts `--replay`, prints deterministic JSON, and exits with the
status its verification objective requires. Lab 8 exits `1` when the fidelity
gate blocks the unit swap; that is the expected result, not a failure of the
driver. `run_all.py` treats that one status as expected.

The book snippets import the workbook facade. Either install the workbook or
put its vendor tree on the path:

```bash
pip install -e workbooks/mjolnir_containment
# or, without installing:
export PYTHONPATH="$PWD/vendor:$PYTHONPATH"
python -c "from mjolnir_containment import Guard, Authority, typed_verdict"
```

Run the set:

```bash
python run_all.py
```

Run the workbook tests:

```bash
python -m pytest tests -q
```

## Vendor provenance

`vendor/` is a source-only copy of the containment libraries from the
research repository:

- `mjolnir_guard`
- `mjolnir_decide`
- `mjolnir_prover`
- `mjolnir_rlvf`
- `mjolnir_intent`
- `mjolnir_common`

The prover's multi-hundred-megabyte `storage/data` corpora, bytecode,
editor files, and pytest caches are deliberately excluded. The libraries are
not imported by `mjolnir_ml`; they live here so the workbook is self-contained
and so Volume I remains untouched.

## Lab map

| Lab | Driver | Chapter |
|---|---|---|
| 1 | `labs/lab01_paralysis.py` | Reactive containment |
| 2 | `labs/lab02_shocks.py` | Belief-space planning |
| 3 | `labs/lab03_horizon.py` | Horizon soundness |
| 4 | `labs/lab04_two_key.py` | Two-key release |
| 5 | `labs/lab05_heads.py` | Calibrated safety heads |
| 6 | `labs/lab06_prover.py` | Claim verification |
| 7 | `labs/lab07_laundering.py` | Decline training |
| 8 | `labs/lab08_fidelity.py` | Commitment fidelity |
| 9 | `labs/lab09_reachability.py` | Reachability |
| 10 | `labs/lab10_matched_weight.py` | Matched-weight control |
| 11 | `labs/lab11_contour.py` | Contour accounting |
| 12 | `labs/lab12_ledger.py` | Boundary ledger |
