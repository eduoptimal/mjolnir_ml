"""The RELEASE gate for a model-backed head: a service that answers a constant
must fail its release test.

The contract selftest cannot see this — it passed 7/7 while `noul` was always 0,
because every check was about the wire, none about the number. This gate runs the
100-state battery the paper's eval used and asserts the properties that make a
number an estimate rather than a constant.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys
import time
import unittest

_REPO = pathlib.Path(__file__).resolve().parents[2]
BATTERY = (_REPO / "mjolnir_research/mjolnir_experiments/inception/outreach/data"
                   "/responses.jsonl")
THRESHOLD = 0.8


def _model():
    try:
        import torch            # noqa: F401
        from transformers import AutoModel  # noqa: F401
    except Exception:                                    # noqa: BLE001
        return None
    from mjolnir_decide.model import Model
    return Model()


class ReleaseGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BATTERY.exists():
            raise unittest.SkipTest(f"battery absent: {BATTERY}")
        cls.model = _model()
        if cls.model is None:
            raise unittest.SkipTest("torch/transformers unavailable here")
        cls.rows = [json.loads(l) for l in open(BATTERY)]
        cls.vals, cls.lat = [], []
        for r in cls.rows:
            t0 = time.perf_counter()
            cls.vals.append(cls.model.noul(r["state"]))
            cls.lat.append((time.perf_counter() - t0) * 1000)

    def test_discriminates_instead_of_answering_a_constant(self):
        vals = set(self.vals)
        released = sum(1 for v in self.vals if float(v) >= THRESHOLD)
        self.assertGreaterEqual(len(vals), 2, "noul is a constant")
        self.assertTrue(0 < released < len(self.rows),
                        f"released={released} of {len(self.rows)}")
        # the eval's number is the reference: released 27 of 100 (band, not a brittle exact)
        self.assertTrue(20 <= released <= 35, f"released={released}, eval said 27")

    def test_we_lax_is_recorded(self):
        lax = [r["id"] for r, v in zip(self.rows, self.vals)
               if float(v) >= THRESHOLD and float(r.get("noul") or 0) < THRESHOLD]
        self.assertLessEqual(len(lax), 12, f"we_lax={len(lax)}: {lax[:6]}")

    def test_determinism(self):
        for i in range(10):
            self.assertEqual(self.model.noul(self.rows[i]["state"]),
                             self.model.noul(self.rows[i]["state"]))

    def test_certification_narrows_never_widens(self):
        """safety-policy@1 on the battery: certified <= head per state, and
        the gate only ever REMOVES releases (measured: head 27, certified 25)."""
        from mjolnir_decide.certify import certify
        rel_head = rel_cert = 0
        for r in self.rows:
            head = self.model.noul(r["state"])
            cert = 0 if certify(r["state"]) else head
            self.assertLessEqual(float(cert), float(head),
                                 f"certification widened: {r['id']}")
            rel_head += float(head) >= THRESHOLD
            rel_cert += float(cert) >= THRESHOLD
        self.assertTrue(20 <= rel_head <= 35, f"head released={rel_head}")
        self.assertTrue(20 <= rel_cert <= rel_head,
                        f"certified released={rel_cert} of head {rel_head}")

    def test_out_of_regime_fails_closed(self):
        ex = self.model.explain("zxqv blorptang " * 40)
        self.assertFalse(ex["in_regime"])
        self.assertEqual(int(ex["noul"]), 0)

    def test_latency_budget(self):
        p50 = statistics.median(self.lat)
        self.assertLess(p50, 6.0, f"p50={p50:.2f} ms")

    def test_provenance_is_pinned(self):
        p = self.model.provenance()
        self.assertTrue(p["checkpoint"]["digest"].startswith("sha256:"))
        self.assertEqual(p["floor"]["quantity"], "flow_score_per_dim")
        self.assertIn("not bit-replayable", p["replay"])
        self.assertEqual(p["blind_spot"]["prose_hazards"], "unread")

    def test_disposition_wrapper(self):
        from mjolnir_decide.disposition import to_disposition
        from mjolnir_decide.serve import answer
        req = {"model": "typesafe/jev-1.13", "state": self.rows[0]["state"],
               "questions": {"safe_to_run": {"type": "noul"}}}
        d = to_disposition(answer(self.model, req))
        self.assertEqual(d["kind"], "estimate")
        self.assertIn(d["verdict"], ("ALLOW", "ASK"))     # never DENY
        # out-of-regime is ASK even if the number alone would allow
        d2 = to_disposition({"answers": {"safe_to_run": {"noul": 1}},
                             "mjolnir": {"in_regime": False}})
        self.assertEqual(d2["verdict"], "ASK")
        self.assertEqual(d2["stage"], "estimated-out-of-regime")


if __name__ == "__main__":
    unittest.main()
