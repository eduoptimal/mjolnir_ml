"""mjolnir_ml.vendors -- CLONES of mjolnir components this package depends on.

STANDALONE RULE: mjolnir_ml imports nothing else from the mjolnir repo at
runtime. Anything it needs lives here as a clone, pinned byte-identical by
the test suite where the clone must not drift, or adapted with the
delta stated in the file header where it must.

    vendors/core/verdict.py  clone of src/mjolnir_router/core/verdict.py
    vendors/wire.py          clone of src/mjolnir_disposition/engine.py (import line adapted)
"""
