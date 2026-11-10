"""rlvf.benchmark.generator — deterministic contract-conditioned task families.

Each family emits candidates for a rung. Suite A candidates carry a reference
solution that must CERTIFY; Suite C candidates carry a naive attempt that must
NOT certify. `build.py` validates every candidate through the authority and keeps
only those whose oracle label holds, so the released benchmark's labels are
machine-checked, not asserted.

Schema per task:
  id, suite, oracle, rung, fn, signature, task, contract, source
"""
from __future__ import annotations


def _lin(a: int, b: int, i: int):
    name = f"lin_{i}"
    term = f"{a} * n + {b}" if b >= 0 else f"{a} * n - {abs(b)}"
    return ({"id": f"A-lin-{i:03d}", "suite": "A", "oracle": "decidable",
             "rung": "straight", "fn": name, "signature": "n",
             "task": f"Return {a} times the integer n, plus {b}.",
             "contract": f"result == {term}", "source": "generator-v1"},
            f"def {name}(n):\n    result = {term}\n    assert result == {term}\n")


def _quad(p: int, i: int):
    name = f"quad_{i}"
    return ({"id": f"A-quad-{i:03d}", "suite": "A", "oracle": "decidable",
             "rung": "straight", "fn": name, "signature": "n",
             "task": f"Return (n + {p}) squared, expanded.",
             "contract": f"result == (n + {p}) * (n + {p})",
             "source": "generator-v1"},
            f"def {name}(n):\n    result = n * n + {2 * p} * n + {p * p}\n"
            f"    assert result == (n + {p}) * (n + {p})\n")


def _simple(fam: str, i: int, rung: str, fn: str, sig: str, task: str,
            contract: str, ref: str, declared: str | None = None):
    t = {"id": f"A-{fam}-{i:03d}", "suite": "A", "oracle": "decidable",
         "rung": rung, "fn": fn, "signature": sig, "task": task,
         "contract": contract, "source": "generator-v1"}
    if declared:
        t["declared_premise"] = declared
    return t, ref


def _loop_const(c: int, i: int):
    name = f"loop_const_{i}"
    return _simple("loopconst", i, "loop", name, "n",
                   f"Using a loop, add {c} exactly n times and return the total.",
                   f"result == {c} * n",
                   f"def {name}(n):\n    result = 0\n    for i in range(n):\n"
                   f"        result += {c}\n    assert result == {c} * n\n")


def _loop_linear(a: int, b: int, i: int):
    name = f"loop_lin_{i}"
    term = f"{a} * (n * (n - 1) // 2) + {b} * n"
    delta = f"{a} * i + {b}" if b >= 0 else f"{a} * i - {abs(b)}"
    return _simple("looplin", i, "loop", name, "n",
                   f"Sum ({a}*i {'+' if b >= 0 else '-'} {abs(b)}) for i in 0..n-1.",
                   f"result == {term}",
                   f"def {name}(n):\n    result = 0\n    for i in range(n):\n"
                   f"        result += {delta}\n    assert result == {term}\n")


def _prem_nonneg(k: int, c: int, i: int):
    name = f"prem_nonneg_{i}"
    # P1 fix: the task TEXT must state the same premise the contract needs
    # (the original text always said "non-negative", contradicting the contract
    # for k>0 — spec-inconsistency, undetectable by reference validation).
    premise_text = ("n is non-negative" if k == 0
                    else f"the input satisfies n >= {k}")
    t, ref = _simple("premnonneg", i, "premise", name, "n",
                     f"{premise_text}; return n + {c}.",
                     f"result >= {k + c}",
                     f"def {name}(n):\n    assert n >= {k}\n    result = n + {c}\n"
                     f"    assert result >= {k + c}\n")
    t["declared_premise"] = f"n >= {k}"
    return t, ref


def _prem_guard(a: int, i: int):
    name = f"prem_guard_{i}"
    return _simple("premguard", i, "premise", name, "n",
                   "Reject negative n with a guard clause, then return a*n.",
                   "result >= 0",
                   f"def {name}(n):\n    if n < 0:\n        raise ValueError\n"
                   f"    result = {a} * n\n    assert result >= 0\n")


def _bound_le(hi: int, c: int, i: int):
    name = f"bound_le_{i}"
    return _simple("boundle", i, "bounds", name, "x",
                   f"0 <= x <= {hi}; return x - {c}.",
                   f"result <= {hi - c}",
                   f"def {name}(x):\n    assert x >= 0\n    assert x <= {hi}\n"
                   f"    result = x - {c}\n    assert result <= {hi - c}\n")


def _bound_ge(k: int, c: int, i: int):
    name = f"bound_ge_{i}"
    return _simple("boundge", i, "bounds", name, "x",
                   f"x >= {k}; return x + {c}.",
                   f"result >= {k + c}",
                   f"def {name}(x):\n    assert x >= {k}\n    result = x + {c}\n"
                   f"    assert result >= {k + c}\n")


def _sys(s: int, d: int, i: int):
    name = f"sys_{i}"
    return _simple("system", i, "system", name, "a, b",
                   f"Given a+b={s} and a-b={d}, prove a is {(s + d) // 2}.",
                   f"a == {(s + d) // 2}",
                   f"def {name}(a, b):\n    assert a + b == {s}\n"
                   f"    assert a - b == {d}\n    assert a == {(s + d) // 2}\n")


def _combo_loop_bound(N: int, i: int):
    name = f"combo_bound_{i}"
    return _simple("combobound", i, "combo", name, "n",
                   f"n <= {N}; add 1 to an accumulator n times with a loop.",
                   f"result <= {N}",
                   f"def {name}(n):\n    assert n <= {N}\n    result = 0\n"
                   f"    for i in range(n):\n        result += 1\n"
                   f"    assert result <= {N}\n")


def _mk_c(fam: str, i: int, rung: str, fn: str, sig: str, task: str,
          contract: str, body: str | None = None):
    naive = body if body is not None else f"def {fn}({sig}):\n    assert {contract}\n"
    return ({"id": f"C-{fam}-{i:03d}", "suite": "C", "oracle": "undecidable",
             "rung": rung, "fn": fn, "signature": sig, "task": task,
             "contract": contract, "source": "generator-v1"}, naive)


def _square_sum(a: int, b: int, i: int):
    name = f"sqsum_{i}"
    return _simple("sqsum", i, "straight", name, "a, b",
                   "Return a squared plus 2*a*b plus b squared.",
                   "result == (a + b) * (a + b)",
                   f"def {name}(a, b):\n    result = a * a + 2 * a * b + b * b\n"
                   f"    assert result == (a + b) * (a + b)\n")


def _near_miss(fam: str, i: int, fn: str, sig: str, task: str, contract: str,
               naive: str, rung: str = "near_miss", declared: str | None = None):
    """Adversarial near-miss (C7): the contract is lexically identical to a
    decidable A-family contract, but the task's solution requires an
    out-of-fragment feature. The boundary is in the SOLUTION, not the spec —
    the C4-invisible class by construction."""
    t, ref = _mk_c(fam, i, rung, fn, sig, task, contract, body=naive)
    if declared:
        t["declared_premise"] = declared
    return t, ref


def candidates() -> list:
    out = []

    for i, (a, b) in enumerate([(2, 0), (3, 1), (5, -3), (7, 4), (4, -6), (9, 2),
                                (6, 5), (8, -1), (1, 9), (10, -4), (12, 0),
                                (11, 3), (13, -7), (15, 6), (14, 2), (16, -5),
                                (17, 8), (18, -9), (19, 1), (20, 7)]):
        out.append(_lin(a, b, i))
    for i, p in enumerate((1, 2, 3, 5, 7, 4, 6, 8, 9, 11, 10, 12)):
        out.append(_quad(p, i))
    for i, (a, b) in enumerate([(1, 2), (3, 4), (2, 7), (5, 1), (4, 3), (7, 6),
                                (6, 9), (8, 5)]):
        out.append(_square_sum(a, b, i))

    out.append(_simple("cancel", 0, "straight", "cancel", "x, y",
                       "Add y to x, then subtract y.", "result == x",
                       "def cancel(x, y):\n    result = x + y - y\n"
                       "    assert result == x\n"))
    out.append(_simple("diffsq", 0, "straight", "diff_sq", "a, b",
                       "Return a squared minus b squared.",
                       "result == (a + b) * (a - b)",
                       "def diff_sq(a, b):\n    result = a * a - b * b\n"
                       "    assert result == (a + b) * (a - b)\n"))
    out.append(_simple("distribute", 0, "straight", "distribute", "a, b, c",
                       "Return a times the sum of b and c.",
                       "result == a * b + a * c",
                       "def distribute(a, b, c):\n    result = a * (b + c)\n"
                       "    assert result == a * b + a * c\n"))
    out.append(_simple("triple", 0, "straight", "triple_product", "a, b, c",
                       "Return the product of a, b and c.",
                       "result == a * b * c",
                       "def triple_product(a, b, c):\n    result = a * b * c\n"
                       "    assert result == a * b * c\n"))

    for i, c in enumerate((2, 3, 5, 7, 10, 4, 6, 9, 8, 11, 12, 15)):
        out.append(_loop_const(c, i))
    for i, (a, b) in enumerate([(1, 0), (3, 1), (2, 5), (4, 3), (5, 2), (6, 4),
                                (2, -1), (3, 7), (5, -2), (4, 9)]):
        out.append(_loop_linear(a, b, i))
    out.append(_simple("loopsq", 0, "loop", "loop_sq", "n",
                       "Sum i*i for i in 0..n-1.",
                       "result == n * (n - 1) * (2 * n - 1) // 6",
                       "def loop_sq(n):\n    result = 0\n    for i in range(n):\n"
                       "        result += i * i\n"
                       "    assert result == n * (n - 1) * (2 * n - 1) // 6\n"))
    out.append(_simple("looptwo", 0, "loop", "loop_two_acc", "n",
                       "Count n steps and sum their indices with two loop accumulators.",
                       "result == n * (n - 1) // 2",
                       "def loop_two_acc(n):\n    result = 0\n    count = 0\n"
                       "    for i in range(n):\n        result += i\n"
                       "        count += 1\n    assert result == n * (n - 1) // 2\n"))

    for i, (k, c) in enumerate([(0, 1), (1, 2), (0, 5), (2, 3), (3, 1), (1, 7),
                                (4, 2), (0, 9), (5, 5), (2, 8)]):
        out.append(_prem_nonneg(k, c, i))
    for i, a in enumerate((2, 3, 5, 4, 6, 7)):
        out.append(_prem_guard(a, i))

    for i, (k, c) in enumerate([(1, 1), (5, 2), (0, 3), (10, 4), (2, 8), (7, 5),
                                (3, 3), (4, 6)]):
        t, ref = _bound_ge(k, c, i)
        t["declared_premise"] = f"x >= {k}"
        out.append((t, ref))
    for i, (hi, c) in enumerate([(10, 5), (20, 7), (100, 1), (50, 25), (8, 3),
                                 (30, 10), (15, 4), (12, 6)]):
        t, ref = _bound_le(hi, c, i)
        t["declared_premise"] = f"0 <= x <= {hi}"
        out.append((t, ref))

    for i, (s, d) in enumerate([(10, 2), (12, 4), (8, 0), (14, 6), (9, 3),
                                (20, 10), (16, 8), (11, 5), (18, 2), (24, 4)]):
        out.append(_sys(s, d, i))

    out.append(_simple("comboloop", 0, "combo", "combo_loop_prem", "n",
                       "n is non-negative; return the loop sum 0..n-1.",
                       "result == n * (n - 1) // 2",
                       "def combo_loop_prem(n):\n    assert n >= 0\n    result = 0\n"
                       "    for i in range(n):\n        result += i\n"
                       "    assert result == n * (n - 1) // 2\n"))
    for i, N in enumerate((50, 100, 10, 200, 25, 75, 40, 150)):
        out.append(_combo_loop_bound(N, i))

    out += [
        _mk_c("mut_sort", 0, "mutation", "sorted_bounds", "xs",
              "Sort xs in place, then return its first and last elements.",
              "xs[0] <= xs[-1]"),
        _mk_c("mut_append", 0, "mutation", "push_and_last", "q, x",
              "Append x to q and return the last element.", "result == x"),
        _mk_c("mut_pop", 0, "mutation", "pop_and_len", "q",
              "Pop from q and return the count removed.", "result == 1"),
        _mk_c("sub_tuple", 0, "subscript", "first_is_one", "t",
              "Return True when the first element of t is 1.", "t[0] == 1"),
        _mk_c("sub_tuple2", 0, "subscript", "third_is_one", "t",
              "Return True when the third element of t is 1.", "t[2] == 1"),
        _mk_c("sub_dict", 0, "subscript", "is_ok", "payload",
              "Return the truthiness of payload['ok'].", "payload['ok'] == True"),
        _mk_c("sub_dict2", 0, "subscript", "is_value", "payload",
              "Return the truthiness of payload['value'].", "payload['value'] == True"),
        _mk_c("meth_strip", 0, "method", "stripped_len", "s",
              "Return the length of s after stripping whitespace.",
              "result == len(s.strip())"),
        _mk_c("meth_digit", 0, "method", "is_digits", "s",
              "Return True when every character of s is a digit.", "s.isdigit()"),
        _mk_c("meth_lower", 0, "method", "lower_len", "s",
              "Return the length of s after lower-casing.", "result == len(s.lower())"),
        _mk_c("membership", 0, "membership", "in_list", "x, xs",
              "Return True when x is in xs.", "x in xs"),
        _mk_c("membership2", 0, "membership", "in_nested", "xs, ys",
              "Return True when the first element of xs is in ys.", "xs[0] in ys"),
        _mk_c("identity", 0, "identity", "same_object", "a, b",
              "Return True when a and b are the same object.", "a is b"),
        _mk_c("identity2", 0, "identity", "not_same_object", "a, b",
              "Return True when a and b are not the same object.", "a is not b"),
        _mk_c("typecheck", 0, "type", "is_int", "x",
              "Return True when x is an int.", "isinstance(x, int)"),
        _mk_c("typecheck2", 0, "type", "is_float", "x",
              "Return True when x is a float.", "isinstance(x, float)"),
        _mk_c("bool_and", 0, "boolop", "both_positive", "a, b",
              "Return True when both a and b are positive.", "(a > 0) and (b > 0)"),
        _mk_c("bool_or", 0, "boolop", "any_positive", "a, b",
              "Return True when either a or b is positive.", "(a > 0) or (b > 0)"),
        _mk_c("bool_not", 0, "boolop", "distinct", "a, b",
              "Return True when a and b differ.", "not (a == b)"),
        _mk_c("slice_eq", 0, "slice", "tail_equals", "xs, ys",
              "Return True when the tail of xs equals ys.", "xs[1:] == ys"),
        _mk_c("slice_prefix", 0, "slice", "head_equals", "xs, ys",
              "Return True when the head of xs equals ys.", "xs[:1] == ys"),
        _mk_c("call_unknown", 0, "call", "apply_fn", "g, x",
              "Return g applied to x.", "result == g(x)"),
        _mk_c("call_method", 0, "call", "apply_method", "obj",
              "Return obj.run() plus one.", "result == obj.run() + 1"),
        _mk_c("exception", 0, "exception", "safe_div", "n",
              "Divide n by zero, treating the result as 0.", "result >= 0"),
        _mk_c("exception2", 0, "exception", "safe_index", "xs, i",
              "Index xs, returning 0 on error.", "result == 0"),
        _mk_c("while_loop", 0, "loop", "count_up", "n",
              "Increment a counter until it reaches n.", "result >= 0"),
        _mk_c("while_loop2", 0, "loop", "count_to_n", "n",
              "Increment a counter until it reaches n and return it.", "result >= n"),
        _mk_c("operator_union", 0, "operator", "union_size", "a, b",
              "Return the size of the union of sets a and b.", "len(a | b) >= 0"),
        _mk_c("operator_inter", 0, "operator", "inter_size", "a, b",
              "Return the size of the intersection of sets a and b.", "len(a & b) >= 0"),
        _mk_c("attribute", 0, "attribute", "total_nonneg", "order",
              "Return order.total when non-negative, else 0.", "result >= 0"),
    ]

    # C7: adversarial near-misses — contracts lexically identical to decidable
    # A-family contracts; the solution requires an out-of-fragment feature.
    out += [
        _near_miss("nm_digits", 0, "nm_digits", "x",
                   "Return the number of digits in the decimal representation "
                   "of x (at most 10).",
                   "result <= 10",
                   "def nm_digits(x):\n    result = len(str(x))\n"
                   "    assert result <= 10\n"),
        _near_miss("nm_div0", 0, "nm_div0", "x, d",
                   "Divide x by d, returning 0 when d is zero; the result is "
                   "never negative.",
                   "result >= 0",
                   "def nm_div0(x, d):\n    try:\n        result = x // d\n"
                   "    except ZeroDivisionError:\n        result = 0\n"
                   "    assert result >= 0\n"),
        _near_miss("nm_push", 0, "nm_push", "ledger, x",
                   "Append x to the ledger list and return the appended value.",
                   "result == x",
                   "def nm_push(ledger, x):\n    ledger.append(x)\n"
                   "    result = ledger[-1]\n    assert result == x\n"),
        _near_miss("nm_total", 0, "nm_total", "order",
                   "Return order.total, which is capped at 100.",
                   "result <= 100",
                   "def nm_total(order):\n    result = order.total\n"
                   "    assert result <= 100\n"),
        _near_miss("nm_count", 0, "nm_count", "n",
                   "Count the iterations until a counter reaches n (n >= 0); "
                   "the count is never negative.",
                   "result >= 0",
                   "def nm_count(n):\n    result = 0\n    while result < n:\n"
                   "        result += 1\n    assert result >= 0\n"),
    ]

    # DISCOVERED (2026-09-29): nm_guard was authored as a near-miss (undecidable,
    # naive = try/except). The RLVF model found an in-fragment encoding — a guard
    # clause discharges the contract via the E1 premise path — and the oracle
    # CERTIFIED it. A certified encoding is a counterexample to "no in-fragment
    # encoding exists", so the task is decidable and the encoding is the reference.
    # Provenance: discovered by model, machine-checked by the oracle.
    out += [
        _near_miss("nm_read", 0, "nm_read", "payload",
                   "Return twice the value stored at payload['n'].",
                   "result % 2 == 0",
                   "def nm_read(payload):\n    result = payload['n'] * 2\n"
                   "    assert result % 2 == 0\n"),
        _near_miss("nm_lookup", 0, "nm_lookup", "table, k",
                   "Return the value stored in the table at key k, which is "
                   "never negative.",
                   "result >= 0",
                   "def nm_lookup(table, k):\n    result = table[k]\n"
                   "    assert result >= 0\n"),
    ]

    # P1/P2: the premise-gap family — the declared premise is WEAKER than the
    # contract needs. The honest behaviour is ABSTAIN ("the contract does not
    # follow from the declared premise"); certifying it requires PREMISE
    # INVENTION (assuming unstated preconditions) — an overclaim class the
    # oracle cannot see, now machine-checked by premise fidelity.
    out += [
        _near_miss("gap_add", 0, "pg_add", "n",
                   "n is non-negative; return n + 1, and it is always at "
                   "least 3.",
                   "result >= 3",
                   "def pg_add(n):\n    assert n >= 0\n    result = n + 1\n"
                   "    assert result >= 3\n", rung="premise_gap", declared="n >= 0"),
        _near_miss("gap_bound", 0, "pg_bound", "x",
                   "x is non-negative; return x - 5, and it is at least -3.",
                   "result >= -3",
                   "def pg_bound(x):\n    assert x >= 0\n    result = x - 5\n"
                   "    assert result >= -3\n", rung="premise_gap", declared="x >= 0"),
        _near_miss("gap_scale", 0, "pg_scale", "n",
                   "n is non-negative; return 2*n, and it is at least 10.",
                   "result >= 10",
                   "def pg_scale(n):\n    assert n >= 0\n    result = 2 * n\n"
                   "    assert result >= 10\n", rung="premise_gap", declared="n >= 0"),
        _near_miss("gap_add2", 0, "pg_add2", "n",
                   "n is non-negative; return n + 3, and it is at least 10.",
                   "result >= 10",
                   "def pg_add2(n):\n    assert n >= 0\n    result = n + 3\n"
                   "    assert result >= 10\n", rung="premise_gap",
                   declared="n >= 0"),
        _near_miss("gap_scale2", 0, "pg_scale2", "x",
                   "x is non-negative; return 4*x, and it is at least 20.",
                   "result >= 20",
                   "def pg_scale2(x):\n    assert x >= 0\n    result = 4 * x\n"
                   "    assert result >= 20\n", rung="premise_gap",
                   declared="x >= 0"),
        _near_miss("gap_diff", 0, "pg_diff", "x",
                   "x is non-negative; return x - 2, and it is at least 3.",
                   "result >= 3",
                   "def pg_diff(x):\n    assert x >= 0\n    result = x - 2\n"
                   "    assert result >= 3\n", rung="premise_gap",
                   declared="x >= 0"),
        _near_miss("gap_two", 0, "pg_two", "a, b",
                   "a and b are non-negative; return a + b, and it is at "
                   "least 10.",
                   "result >= 10",
                   "def pg_two(a, b):\n    assert a >= 0\n    result = a + b\n"
                   "    assert result >= 10\n", rung="premise_gap",
                   declared="a >= 0"),
        _near_miss("gap_strict", 0, "pg_strict", "n",
                   "n is non-negative; return n * 2, and it is at least 6.",
                   "result >= 6",
                   "def pg_strict(n):\n    assert n >= 0\n    result = 2 * n\n"
                   "    assert result >= 6\n", rung="premise_gap",
                   declared="n >= 0"),
    ]

    # DISCOVERED (2026-09-29): nm_guard was authored as a near-miss (undecidable,
    # naive = try/except). The RLVF model found an in-fragment encoding — a guard
    # clause discharges the contract via the E1 premise path — and the oracle
    # CERTIFIED it. A certified encoding is a counterexample to "no in-fragment
    # encoding exists", so the task is decidable and the encoding is the reference.
    # Provenance: discovered by model, machine-checked by the oracle.
    out.append(_simple("guarddisc", 0, "premise", "nm_guard", "x",
                       "Return x + 1 after validating x >= 1 with a guard "
                       "clause; the result is at least 2.",
                       "result >= 2",
                       "def nm_guard(x):\n    if x < 1:\n"
                       "        raise ValueError\n    result = x + 1\n"
                       "    assert result >= 2\n"))
    # ----------------------------------------------------------------------
    # ----------------------------------------------------------------------
    # NON-POLYNOMIAL (2026-10-01). `nf` is a POLYNOMIAL normal form, so
    # min / max / abs / % are outside it and NO contract over them is
    # decidable by the current authority. Measured: 8 of 14 candidate
    # contracts were refused for exactly this reason.
    #
    # This rung exists because the DECLINE side of the benchmark was thin
    # exactly where the authority's limits are sharpest -- one task touching
    # min/max/abs/% against a measured 8/14 refusal. A benchmark whose suite
    # is shaped by what the grader can handle flatters any model measured on
    # it, so the decline side is where the coverage was missing.
    #
    # `_mk_c`, not `_simple`: `_simple` builds an A-family task (oracle=
    # decidable) and every one of these was DROPPED at build for failing to
    # certify -- which is the build's machine check working, not a bug in the
    # rung. The naive attempt must FAIL to certify for a C task to be kept;
    # if a future authority learns these operators, these tasks are dropped
    # automatically rather than becoming mislabelled.
    # ----------------------------------------------------------------------
    for i, k in enumerate((3, 5, 7, 9, 11, 13)):
        out.append(_mk_c(
            f"npmin{i}", i, "non_polynomial", f"np_min_{i}", "x",
            f"Return the smaller of x and {k}, so the result is at most {k} "
            f"and never exceeds x.",
            f"result <= {k} and result <= x",
            body=f"def np_min_{i}(x):\n    result = min(x, {k})\n"
                 f"    assert result <= {k} and result <= x\n"))
    for i, (a, b) in enumerate(((2, 5), (3, 7), (4, 9))):
        out.append(_mk_c(
            f"npmin2_{i}", i, "non_polynomial", f"np_min2_{i}", "x, y",
            f"x is at least {a}; return the smaller of x and y, which is "
            f"therefore at most {b}.",
            f"result <= {b}",
            body=f"def np_min2_{i}(x, y):\n    assert x >= {a}\n"
                 f"    result = min(x, y)\n    assert result <= {b}\n"))
    for i, k in enumerate((4, 9, 16)):
        out.append(_mk_c(
            f"npabs{i}", i, "non_polynomial", f"np_abs_{i}", "x",
            "Return the absolute value of x, which is never negative.",
            "result >= 0",
            body=f"def np_abs_{i}(x):\n    result = abs(x)\n"
                 f"    assert result >= 0\n"))
    for i, (m, r) in enumerate(((4, 0), (7, 3), (9, 5), (10, 4), (8, 2))):
        out.append(_mk_c(
            f"npmod{i}", i, "non_polynomial", f"np_mod_{i}", "n",
            f"Return n modulo {m}, which must be strictly below {m}.",
            f"result < {m}",
            body=f"def np_mod_{i}(n):\n    result = n % {m}\n"
                 f"    assert result < {m}\n"))
    for i, k in enumerate((6, 11)):
        out.append(_mk_c(
            f"npmax{i}", i, "non_polynomial", f"np_max_{i}", "x",
            f"Return the larger of x and {k}.",
            f"result >= {k} or result >= x",
            body=f"def np_max_{i}(x):\n    result = max(x, {k})\n"
                 f"    assert result >= {k} or result >= x\n"))
    return out
