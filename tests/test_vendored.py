"""Clone integrity: mjolnir_ml stays standalone and its clones do not drift."""
from pathlib import Path

import mjolnir_ml

ROOT = Path(__file__).resolve().parents[3]


def test_core_clone_is_byte_identical():
    src = ROOT / "src/mjolnir_router/core/verdict.py"
    if not src.exists():
        import pytest
        pytest.skip("repo source not present (standalone checkout)")
    vendored = Path(mjolnir_ml.__file__).parent / "vendors/core/verdict.py"
    assert vendored.read_bytes() == src.read_bytes(), \
        "vendored core drifted from src/mjolnir_router/core/verdict.py"


def test_wire_clone_only_changed_the_import():
    src = ROOT / "src/mjolnir_disposition/engine.py"
    if not src.exists():
        import pytest
        pytest.skip("repo source not present (standalone checkout)")
    vendored = Path(mjolnir_ml.__file__).parent / "vendors/wire.py"
    text = vendored.read_text()
    assert "from mjolnir_ml.vendors.core.verdict import" in text
    assert "from mjolnir_router" not in text
    # everything after the import block is byte-identical
    def body(t):
        lines = t.splitlines()
        start = next(i for i, ln in enumerate(lines)
                     if ln.startswith("RELEASE ="))
        return "\n".join(lines[start:])
    assert body(text) == body(src.read_text())


def test_no_repo_imports_at_runtime():
    pkg = Path(mjolnir_ml.__file__).parent
    for path in pkg.rglob("*.py"):
        for ln in path.read_text().splitlines():
            if ln.startswith(("from mjolnir_router", "import mjolnir_router",
                              "from mjolnir_disposition",
                              "import mjolnir_disposition")):
                raise AssertionError(f"{path}: repo dependency {ln!r}")
