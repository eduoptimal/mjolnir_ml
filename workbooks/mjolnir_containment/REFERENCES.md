# Volume reference boundaries

The workbook is additive. It does not modify `mjolnir_ml`, its tests, or its
packaging.

| Volume | Code | Import rule |
|---|---|---|
| Volume I — *Fail-Closed Machine Learning* | `mjolnir_ml` package | The workbook does not import it at package import time. Volume I primitives are reached only through `mjolnir_containment.integrations`. |
| Volume II — *Non-Stationary Machine Learning* | Volume II modules in `mjolnir_ml` (`calibration`, `pair`, `hazards`, `readiness`, `live`, `discovery`, `causal`, `regime`) | Same rule: explicit `integrations` adapters only. |
| Volume III — *Agentic Machine Learning* | `workbooks/mjolnir_containment` | The book snippets import `mjolnir_containment`; the implementation packages are vendored under `workbooks/mjolnir_containment/vendor`. |

## Import surface

```bash
cd workbooks/mjolnir_containment
export PYTHONPATH="$PWD/vendor:$PYTHONPATH"
python -c "from mjolnir_containment import Guard, Authority, typed_verdict"
```

The book snippets use:

```python
from mjolnir_containment import Guard, Authority, typed_verdict
```

and the spy harness uses:

```python
from mjolnir_containment import release, DeniedFixtures, Spy, ReaderRevisions
```

## Isolation checks

- `mjolnir_containment` imports only the standard library.
- `mjolnir_containment.integrations` is the only file that names `mjolnir_ml`.
- `workbooks/mjolnir_containment/conftest.py` keeps vendored tests out of the
  Volume I test run.
- `workbooks/mjolnir_containment/vendor` contains no `mjolnir_ml` copy.
