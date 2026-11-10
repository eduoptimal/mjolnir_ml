# mjolnir.storage — persistence layer (module inside the package)

mjolnir generates **artifacts** (mined candidates, confirmation ledgers, corpora,
theory tables, logs, certificates). Persistence is abstracted behind `Store` so
the same engine runs in three modes without change:

| stage | backend | location |
|---|---|---|
| **now** (development / local library) | JSON files on disk | `mjolnir/storage/data/` |
| **next** | Google Cloud Storage bucket | `store://gcs/...` |
| **later** | centralized database / service (mjolnir as a **client**) | `db://<dsn>` |

Callers use `get_store()` and never touch files. `mjolnir.lib.paths` maps data
namespaces onto the same root, so the engine is backend-agnostic.

## Files

| file | role |
|---|---|
| `interface.py` | abstract `Store` (bytes + JSON/JSONL helpers + append + list; namespaced POSIX keys) |
| `local.py` | `LocalStore` — filesystem backend (atomic write, append preserved) |
| `cloud.py` | `CloudStore` — bucket backend (planned) |
| `db.py` | `DbStore` — centralized DB client (planned, the end state) |
| `config.py` | `get_store()` — resolves `MJ_STORAGE` (path / `store://` / `db://`) |
| `data/` | local JSON data root (`registry/ corpus/ theories/ models/ logs/ out/`) |

## Data namespaces (`data/…`)

| path | contents | write model |
|---|---|---|
| `registry/` | mined candidates `mined_*.jsonl`, confirmation ledger, signing key | **append-only** ledger; mined files regenerated |
| `corpus/` | frozen corpora (`fragment_v1.jsonl`, `lean_golden_v1.jsonl`) | frozen, reviewed re-freeze only |
| `theories/` | hash-pinned theory tables (`classification.json`, `intervals.json`) | frozen data |
| `models/` | proposer checkpoints (`ranker-mini/`, copied) | frozen |
| `logs/`, `out/` | run logs, emitted artifacts | overwritten per run |

## Migrating to the bucket, then the DB

1. Implement `CloudStore(Store)` in `cloud.py` (object keys, paginated `list`,
   append via segment objects or a write-ahead object); select it in `config.py`.
2. Load the bucket JSONs into the central DB, then implement `DbStore(Store)` in
   `db.py` (keys → rows/objects; `append_jsonl` → insert/append; `list` → query)
   and set `MJ_STORAGE=db://<dsn>`.
3. The read path stays deterministic and offline-cacheable; the ledger stays
   **append-only** in every backend.

## Discipline

Never store raw source code, SQL, or customer data here — only **shapes**:
hashes, predicates, provenance (repo/commit/line), verdicts. Signed knowledge is
versioned by shape/format digest.
