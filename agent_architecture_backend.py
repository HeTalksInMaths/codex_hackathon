"""Python backend port of the CharDegreeHarnessV2 core logic.

This module keeps the benchmark pipeline from the React/TypeScript harness while
removing UI concerns so it can run in plain Python scripts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Optional

GroupId = tuple[int, int]
SolverStyle = Literal["BY_GROUP_ID", "GLOBAL_SET_OF_SETS"]
UnlockPolicy = Literal["STRICT_BY_ID", "GLOBAL_SET_OF_SETS"]


TICKET_CLASSES: tuple[str, ...] = (
    "OK",
    "PARSE_ERROR",
    "N_MISMATCH",
    "SPEC_LOCK_MISMATCH",
    "DUPLICATE_IDS",
    "MISSING_GROUPS",
    "EXTRA_GROUPS",
    "NON_POSINT_DEGREES",
    "MISSING_DEGREE_1",
    "SUMSQ_VIOLATION",
    "DIVIDES_VIOLATION",
    "LEN_VS_CONJ_CLASSES",
    "COUNT1_VS_ABELIANIZATION",
    "WRONG_DEGREES",
    "WRONG_DISTINCT_SET",
    "WRONG_GLOBAL_SET_OF_SETS",
)


@dataclass
class Failure:
    kind: str
    id: Optional[GroupId] = None
    expected: Optional[list[int]] = None
    got: Optional[Any] = None
    detail: Optional[str] = None


@dataclass
class ParseResult:
    ok: bool
    value: Optional[Any] = None
    error: Optional[str] = None


def fmt(x: Any) -> str:
    return json.dumps(x, indent=2, sort_keys=True)


def stable_stringify(x: Any) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def deep_eq(a: Any, b: Any) -> bool:
    return stable_stringify(a) == stable_stringify(b)


def try_parse_json(text: str) -> ParseResult:
    try:
        return ParseResult(ok=True, value=json.loads(text))
    except json.JSONDecodeError as exc:
        return ParseResult(ok=False, error=str(exc))


def id_to_tuple(raw_id: Any) -> GroupId:
    if not isinstance(raw_id, (list, tuple)) or len(raw_id) != 2:
        raise ValueError(f"invalid id format: {raw_id!r}")
    a, b = raw_id
    if not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int) or isinstance(b, bool):
        raise ValueError(f"id entries must be ints: {raw_id!r}")
    return (a, b)


def key_of_id(group_id: Any) -> str:
    a, b = id_to_tuple(group_id)
    return f"{a}:{b}"


def is_pos_int_array(xs: Any) -> bool:
    return isinstance(xs, list) and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in xs)


def normalize_multiset(xs: list[int]) -> list[int]:
    return sorted(xs)


def multiset_eq(a: list[int], b: list[int]) -> bool:
    return normalize_multiset(a) == normalize_multiset(b)


def distinct_sorted(xs: list[int]) -> list[int]:
    return sorted(set(xs))


def sumsq(xs: Iterable[int]) -> int:
    return sum(d * d for d in xs)


def count_eq(xs: Iterable[int], a: int) -> int:
    return sum(1 for x in xs if x == a)


def sig_of_set(xs: list[int]) -> str:
    return ",".join(str(x) for x in xs)


def unique_degree_sets(sets_: list[list[int]]) -> list[list[int]]:
    m: dict[str, list[int]] = {}
    for s in sets_:
        d = distinct_sorted(s)
        m[sig_of_set(d)] = d
    return sorted(m.values(), key=lambda s: (len(s), s))


def diff_degree_set_lists(oracle: list[list[int]], solver: list[list[int]]) -> dict[str, list[list[int]]]:
    o = {sig_of_set(s): s for s in unique_degree_sets(oracle)}
    s = {sig_of_set(t): t for t in unique_degree_sets(solver)}

    missing = [v for k, v in o.items() if k not in s]
    extra = [v for k, v in s.items() if k not in o]
    return {
        "missing": sorted(missing, key=lambda x: (len(x), x)),
        "extra": sorted(extra, key=lambda x: (len(x), x)),
    }


def find_duplicate_ids(items: list[dict[str, Any]]) -> list[GroupId]:
    seen: set[str] = set()
    dups: list[GroupId] = []
    for item in items:
        try:
            key = key_of_id(item.get("id"))
            group_id = id_to_tuple(item.get("id"))
        except Exception:
            continue
        if key in seen:
            dups.append(group_id)
        else:
            seen.add(key)
    return dups


def failure_to_dict(f: Failure) -> dict[str, Any]:
    d = asdict(f)
    return {k: v for k, v in d.items() if v is not None}


def parse_targets_json(text: str) -> dict[str, Any]:
    p = try_parse_json(text)
    if not p.ok:
        raise ValueError(f"Targets JSON parse error: {p.error}")

    v = p.value
    if not isinstance(v, dict):
        raise ValueError("Targets JSON must be an object")
    if not isinstance(v.get("n"), int) or not isinstance(v.get("nr_groups"), int) or not isinstance(v.get("targets"), list):
        raise ValueError("Targets JSON missing fields: n, nr_groups, targets[]")

    for t in v["targets"]:
        if not isinstance(t, dict):
            raise ValueError("Targets: each target must be an object")
        id_to_tuple(t.get("id"))
        if not isinstance(t.get("order"), int):
            raise ValueError("Targets: each target must include order:number")

    return v


def parse_solver_json(text: str, solver_style: SolverStyle) -> dict[str, Any]:
    p = try_parse_json(text)
    if not p.ok:
        raise ValueError(f"Solver JSON parse error: {p.error}")

    v = p.value
    if not isinstance(v, dict) or not isinstance(v.get("n"), int):
        raise ValueError("Solver JSON missing field: n")
    if "spec_lock" not in v:
        raise ValueError("Solver JSON missing field: spec_lock (must echo locked spec)")

    if solver_style == "BY_GROUP_ID":
        if not isinstance(v.get("answers"), list):
            raise ValueError("Solver JSON (BY_GROUP_ID) missing field: answers[]")
    else:
        if not isinstance(v.get("distinct_degree_sets"), list):
            raise ValueError("Solver JSON (GLOBAL_SET_OF_SETS) missing field: distinct_degree_sets[]")
    return v


def parse_oracle_json(text: str) -> dict[str, Any]:
    p = try_parse_json(text)
    if not p.ok:
        raise ValueError(f"Oracle JSON parse error: {p.error}")

    v = p.value
    if not isinstance(v, dict) or not isinstance(v.get("n"), int) or not isinstance(v.get("oracle"), list):
        raise ValueError("Oracle JSON missing fields: n, oracle[]")
    return v


def build_gap_targets_snippet_v2(
    n: int,
    mode: Literal["FIRST_K", "RANDOM_K", "ALL"] = "FIRST_K",
    take_k: int = 10,
    seed: int = 42,
    include_desc: bool = True,
    include_invariants: bool = True,
) -> str:
    k = max(1, min(int(take_k), 1000))
    seed = max(0, int(seed))

    fields = ["id := [n,i]", "order := Size(G)", "is_abelian := IsAbelian(G)"]
    if include_desc:
        fields.append("desc := StructureDescription(G)")
    if include_invariants:
        fields.extend(
            [
                "nr_conj_classes := NrConjugacyClasses(G)",
                "center_order := Size(Center(G))",
                "abelianization_order := Size(Abelianization(G))",
            ]
        )

    mode_comment = {
        "ALL": "# mode: ALL groups of order n",
        "RANDOM_K": "# mode: RANDOM sample of K groups (seeded)",
        "FIRST_K": "# mode: FIRST K groups (SmallGroup ids 1..K)",
    }[mode]

    rec_body = ",\n    ".join(fields)
    include_desc_s = "true" if include_desc else "false"
    include_inv_s = "true" if include_invariants else "false"

    return f"""# GAP: export TARGETS as JSON (robust; no character degrees)
# Paste into GAP (local) or CoCalc GAP and run.
# Requires GAP package \"json\".
LoadPackage(\"json\");

{mode_comment}
n := {n};
nr := NrSmallGroups(n);
ids := [1..nr];

K := Minimum(Length(ids), {k});
seed := {seed};

if \"{mode}\" = \"ALL\" then
  # keep ids = [1..nr]
elif \"{mode}\" = \"RANDOM_K\" then
  SetSeed(seed);
  Shuffle(ids);
  ids := ids{{[1..K]}};
else
  ids := ids{{[1..K]}};
fi;

targets := List(ids, function(i)
  local G;
  G := SmallGroup(n,i);
  return rec(
    {rec_body}
  );
end);

Print(GapToJsonString(rec(
  n := n,
  nr_groups := nr,
  targets := targets,
  meta := rec(
    generator := \"gap_targets_v2\",
    mode := \"{mode}\",
    K := K,
    seed := seed,
    include_desc := {include_desc_s},
    include_invariants := {include_inv_s}
  )
)));
"""


def build_gap_oracle_snippet_v2(targets: dict[str, Any]) -> str:
    ids = [id_to_tuple(t["id"])[1] for t in targets.get("targets", [])]
    list_i = ",".join(str(i) for i in ids)
    return f"""# GAP: export ORACLE degrees (ground truth) as JSON
# IMPORTANT: do NOT paste oracle output into the LLM. This is the answer key.
LoadPackage(\"json\");

n := {targets['n']};
ids := [{list_i}];

oracle := List(ids, function(i)
  local tbl, degs;
  tbl := CharacterTable(SmallGroup(n,i));
  degs := List(Irr(tbl), chi -> chi[1]);
  return rec(id := [n,i], degrees := degs);
end);

Print(GapToJsonString(rec(
  n := n,
  oracle := oracle,
  meta := rec(generator := \"gap_oracle_v2\")
)));
"""


def build_spec_lock(n: int, solver_style: SolverStyle) -> dict[str, Any]:
    return {
        "task": "distinct_degree_sets_for_order_n" if solver_style == "GLOBAL_SET_OF_SETS" else "char_degrees_by_group_id",
        "domain": "GAP.SmallGroup(n,i)",
        "n": n,
        "output": "global_set_of_sets_distinct_degrees" if solver_style == "GLOBAL_SET_OF_SETS" else "per_group_multiset_degrees",
        "constraints": [
            "degrees_are_positive_integers",
            "degrees_include_1",
            "sumsq_equals_group_order (when per-group)",
            "each_degree_divides_group_order (when per-group)",
        ],
        "version": "v2",
    }


def _safe_target_fields_for_prompt(targets: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in targets.get("targets", []):
        row = {
            "id": t.get("id"),
            "order": t.get("order"),
            "is_abelian": t.get("is_abelian"),
        }
        for k in ("desc", "nr_conj_classes", "center_order", "abelianization_order"):
            if k in t and t[k] is not None:
                row[k] = t[k]
        rows.append(row)
    return rows


def build_solver_prompt_v2(targets: dict[str, Any], spec_lock: dict[str, Any], solver_style: SolverStyle) -> str:
    n = targets["n"]
    schema = (
        {
            "n": n,
            "spec_lock": spec_lock,
            "distinct_degree_sets": [[1], [1, 2], [1, 3]],
            "notes": "optional",
        }
        if solver_style == "GLOBAL_SET_OF_SETS"
        else {
            "n": n,
            "spec_lock": spec_lock,
            "answers": [{"id": [n, 1], "degrees": [1, 1, 2, 2, 2, 2], "notes": "optional per group"}],
            "notes": "optional",
        }
    )

    task_block = (
        [
            "Task:",
            f"Return the SET of SETS of DISTINCT irreducible complex character degrees for ALL targets (order n={n}).",
            "You are NOT required to map sets to specific group ids.",
            "Each inner set must be sorted ascending with no duplicates.",
        ]
        if solver_style == "GLOBAL_SET_OF_SETS"
        else [
            "Task:",
            "For each target group id [n,i], output the MULTISET of irreducible complex character degrees (with multiplicity).",
            "Order does not matter; duplicates matter.",
        ]
    )

    parts = [
        "You are solving a benchmark about finite groups from the GAP SmallGroup(n,i) library.",
        "Return ONLY valid JSON. No markdown, no extra text.",
        "",
        *task_block,
        "",
        "Locked spec (MUST echo exactly):",
        fmt(spec_lock),
        "",
        "Targets (safe invariants only; NO oracle degrees):",
        fmt({"n": n, "nr_groups": targets.get("nr_groups"), "targets": _safe_target_fields_for_prompt(targets)}),
        "",
        "Output schema:",
        fmt(schema),
        "",
        "Hard rules:",
        "- All degrees must be POSITIVE integers.",
        "- Degrees must include 1.",
        "- If output is per-group: sum(d^2) MUST equal |G| for each group.",
        "- If output is per-group: each degree d MUST divide |G|.",
        "",
        "Meta rules:",
        "- Do NOT ask for oracle degrees.",
        "- Do NOT change the schema.",
        "- If uncertain, make a guess BUT it must satisfy the hard rules. (Invalid certificates are rejected.)",
    ]
    return "\n".join(parts)


def build_pre_pass_repair_prompt(spec_lock: dict[str, Any], targets: dict[str, Any], solver: dict[str, Any], precheck: dict[str, Any]) -> str:
    return "\n".join(
        [
            "You previously produced solver output for a character degree benchmark.",
            "Your output failed pre-checks (certificate constraints) BEFORE oracle comparison.",
            "Fix your JSON so it satisfies the constraints exactly.",
            "Return ONLY valid JSON (same schema).",
            "",
            "Locked spec:",
            fmt(spec_lock),
            "",
            "Targets:",
            fmt({"n": targets.get("n"), "targets": targets.get("targets", [])}),
            "",
            "Your previous output:",
            fmt(solver),
            "",
            "Precheck failures (no oracle used):",
            fmt(precheck.get("failures", [])),
            "",
            "Now return a corrected JSON output that passes the constraints.",
        ]
    )


def build_post_pass_proof_harvest_prompt(
    n: int,
    targets: dict[str, Any],
    solver_style: SolverStyle,
    solver_obj: dict[str, Any],
    unlock_policy: UnlockPolicy,
    oracle_report: dict[str, Any],
) -> str:
    _ = oracle_report  # reserved for future use
    schema = {
        "n": n,
        "degree_set_plans": [
            {
                "distinct_degrees": [1, 2],
                "occurs_in_how_many_targets": 3,
                "strategy": "string",
                "steps": [
                    {
                        "kind": "A|B|C",
                        "claim": "string",
                        "lean_lemma_needed": "optional string",
                        "notes": "optional string",
                    }
                ],
            }
        ],
        "lean_backlog": [
            {
                "lemma_name": "string",
                "statement_in_english": "string",
                "priority": "P0|P1|P2",
            }
        ],
    }

    return "\n".join(
        [
            "You just matched the GAP oracle for a character-degree benchmark.",
            "Now we want to HARVEST a proof plan that can be formalized in Lean (eventually in mathlib).",
            "",
            "Return ONLY JSON (no markdown).",
            "",
            f"n = {n}",
            f"solver_style = {solver_style}",
            f"unlock_policy = {unlock_policy}",
            "",
            "Targets (invariants you were given):",
            fmt({"n": targets.get("n"), "targets": targets.get("targets", [])}),
            "",
            "Your correct answer (keep it as-is):",
            fmt(solver_obj),
            "",
            "Your job:",
            "- For each DISTINCT degree-set that occurs, propose a derivation strategy.",
            "- Prefer short, checkable steps: e.g. abelian => all degrees 1; direct product => degrees multiply; standard families (dihedral/quaternion); etc.",
            "- If you used a classification theorem, name it precisely.",
            "- Tag each step as (A) already checkable by arithmetic, (B) checkable once lemma X exists in Lean, or (C) requires bigger theory.",
            "",
            "JSON schema to return:",
            fmt(schema),
        ]
    )


def build_lean_cert_snippet_v2(n: int, items: list[dict[str, Any]]) -> str:
    item_rows: list[str] = []
    for it in items:
        gid = id_to_tuple(it["id"])
        degrees = ", ".join(str(d) for d in it["degrees"])
        extra = ""
        if isinstance(it.get("nr_conj_classes"), int):
            extra += f", nrConjClasses := some {it['nr_conj_classes']}"
        if isinstance(it.get("abelianization_order"), int):
            extra += f", abelianizationOrder := some {it['abelianization_order']}"
        item_rows.append(f"{{ n := {gid[0]}, i := {gid[1]}, order := {it['order']}, degrees := [{degrees}]{extra} }}")

    items_block = ",\n    ".join(item_rows)

    return f"""import Std

open Std

namespace Cert

def sumsq (xs : List Nat) : Nat :=
  xs.foldl (fun acc d => acc + d*d) 0

def countEq (xs : List Nat) (a : Nat) : Nat :=
  xs.foldl (fun acc x => if x = a then acc + 1 else acc) 0

structure Item where
  n : Nat
  i : Nat
  order : Nat
  degrees : List Nat
  nrConjClasses : Option Nat := none
  abelianizationOrder : Option Nat := none

def violations (it : Item) : List String :=
  let mut v : List String := []

  if it.degrees.isEmpty then
    v := v.concat \"degrees list must be nonempty\"

  if it.degrees.any (fun d => d = 0) then
    v := v.concat \"degree must be positive\"

  if not (it.degrees.any (fun d => d = 1)) then
    v := v.concat \"degrees must include 1\"

  let s := sumsq it.degrees
  if s != it.order then
    v := v.concat s!\"sum(d^2) must equal |G| (got {{s}})\"

  if it.degrees.any (fun d => it.order % d != 0) then
    v := v.concat \"each degree must divide |G|\"

  match it.nrConjClasses with
  | some k =>
      if it.degrees.length != k then
        v := v.concat s!\"#degrees must equal #conjugacy classes (expected {{k}})\"
  | none => pure ()

  match it.abelianizationOrder with
  | some m =>
      let c1 := countEq it.degrees 1
      if c1 != m then
        v := v.concat s!\"#(degree=1) must equal |G/G'| (expected {{m}}, got {{c1}})\"
  | none => pure ()

  v

structure Row where
  n : Nat
  i : Nat
  ok : Bool
  sumsq : Nat
  order : Nat
  violations : List String

def row (it : Item) : Row :=
  let s := sumsq it.degrees
  let v := violations it
  {{ n := it.n, i := it.i, ok := v.isEmpty, sumsq := s, order := it.order, violations := v }}

#eval
  let items : List Item := [
    {items_block}
  ]
  let rows := items.map row
  let rowJson (r : Row) : String :=
    "{{" ++
      "\\\"id\\\":[" ++ toString r.n ++ "," ++ toString r.i ++ "]," ++
      "\\\"ok\\\":" ++ (if r.ok then "true" else "false") ++ "," ++
      "\\\"sumsq\\\":" ++ toString r.sumsq ++ "," ++
      "\\\"order\\\":" ++ toString r.order ++ "," ++
      "\\\"violations\\\":[" ++ String.intercalate "," (r.violations.map (fun s => "\\\"" ++ s ++ "\\\"")) ++ "]" ++
    "}}"
  let body := String.intercalate "," (rows.map rowJson)
  "{{\\\"n\\\":" ++ toString {n} ++ ",\\\"results\\\":[" ++ body ++ "]}}"

end Cert
"""


def build_aristotle_howto_snippet() -> str:
    return """# Aristotle (Harmonic) - optional post-pass step
# Goal: help formalize Lean proof skeletons / fill in `sorry` holes.

# 1) Get access / API key:
#   https://aristotle.harmonic.fun/

# 2) Install CLI (Python >= 3.10):
pip install aristotlelib

# 3) Set your API key:
export ARISTOTLE_API_KEY=\"your-api-key-here\"

# 4) Ask Aristotle to fill sorries in a Lean file:
aristotle prove-from-file path/to/File.lean --output-file solution.lean

# Tip:
#   Put an English proof sketch in a comment tagged \"PROVIDED SOLUTION:\" above the theorem.
"""


def _check_spec_lock_or_fail(failures: list[Failure], solver_obj: dict[str, Any], spec_lock: dict[str, Any]) -> None:
    if "spec_lock" not in solver_obj:
        failures.append(Failure(kind="SPEC_LOCK_MISMATCH", detail="missing solver.spec_lock"))
        return
    if not deep_eq(solver_obj["spec_lock"], spec_lock):
        failures.append(Failure(kind="SPEC_LOCK_MISMATCH", detail="solver.spec_lock did not match locked spec"))


def run_precheck(
    targets_obj: dict[str, Any],
    solver_obj: dict[str, Any],
    solver_style: SolverStyle,
    spec_lock: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if spec_lock is None:
        spec_lock = build_spec_lock(int(targets_obj["n"]), solver_style)

    failures: list[Failure] = []

    if solver_obj.get("n") != targets_obj.get("n"):
        failures.append(
            Failure(
                kind="N_MISMATCH",
                detail=f"targets.n={targets_obj.get('n')} but solver.n={solver_obj.get('n')}",
            )
        )

    _check_spec_lock_or_fail(failures, solver_obj, spec_lock)

    target_keys = {key_of_id(t["id"]) for t in targets_obj.get("targets", [])}

    if solver_style == "GLOBAL_SET_OF_SETS":
        raw_sets = solver_obj.get("distinct_degree_sets")
        derived_sets: list[list[int]] = []

        if not isinstance(raw_sets, list):
            failures.append(Failure(kind="PARSE_ERROR", detail="distinct_degree_sets must be an array"))
            raw_count = 0
        else:
            raw_count = len(raw_sets)
            for raw in raw_sets:
                if not is_pos_int_array(raw):
                    failures.append(Failure(kind="NON_POSINT_DEGREES", detail="each distinct_degree_set must be positive integers"))
                    continue
                d = distinct_sorted(raw)
                derived_sets.append(d)
                if len(d) == 0:
                    failures.append(Failure(kind="PARSE_ERROR", detail="distinct_degree_set must be nonempty"))
                if 1 not in d:
                    failures.append(Failure(kind="MISSING_DEGREE_1", detail=f"distinct_degree_set {fmt(d)} missing 1"))

        return {
            "pass": len(failures) == 0,
            "n": targets_obj["n"],
            "style": solver_style,
            "summary": {
                "targets": len(targets_obj.get("targets", [])),
                "answered": raw_count,
                "missing": 0,
                "extra": 0,
                "duplicate_ids": 0,
                "groups_with_violations": len(failures),
            },
            "failures": [failure_to_dict(f) for f in failures],
            "derived": {"solver_distinct_degree_sets": unique_degree_sets(derived_sets)},
        }

    answers = solver_obj.get("answers")
    if not isinstance(answers, list):
        failures.append(Failure(kind="PARSE_ERROR", detail="answers must be an array"))
        answers = []

    dups = find_duplicate_ids(answers)
    if dups:
        failures.append(Failure(kind="DUPLICATE_IDS", detail=f"duplicate ids: {fmt([list(x) for x in dups])}"))

    answered_by_id: dict[str, Any] = {}
    for a in answers:
        if isinstance(a, dict) and "id" in a:
            try:
                answered_by_id[key_of_id(a["id"])] = a.get("degrees")
            except Exception:
                failures.append(Failure(kind="PARSE_ERROR", detail=f"invalid answer id format: {a.get('id')!r}"))

    for t in targets_obj.get("targets", []):
        k = key_of_id(t["id"])
        if k not in answered_by_id:
            failures.append(Failure(id=id_to_tuple(t["id"]), kind="MISSING_GROUPS"))

    for a in answers:
        if not isinstance(a, dict) or "id" not in a:
            continue
        try:
            gid = id_to_tuple(a["id"])
            k = key_of_id(gid)
        except Exception:
            continue
        if k not in target_keys:
            failures.append(Failure(id=gid, kind="EXTRA_GROUPS", detail="answer id not in target set"))

    groups_with_violations = 0
    solver_distinct_sets_all: list[list[int]] = []

    for t in targets_obj.get("targets", []):
        gid = id_to_tuple(t["id"])
        k = key_of_id(gid)
        got = answered_by_id.get(k)
        if got is None:
            continue

        violated_here = False

        if not is_pos_int_array(got):
            failures.append(Failure(id=gid, kind="NON_POSINT_DEGREES", got=got, detail="degrees must be positive integers"))
            violated_here = True
        else:
            if 1 not in got:
                failures.append(Failure(id=gid, kind="MISSING_DEGREE_1", got=got, detail="degrees must include 1"))
                violated_here = True

            s = sumsq(got)
            if s != t["order"]:
                failures.append(Failure(id=gid, kind="SUMSQ_VIOLATION", got=got, detail=f"sum(d^2)={s} != |G|={t['order']}"))
                violated_here = True

            bad_div = next((d for d in got if t["order"] % d != 0), None)
            if bad_div is not None:
                failures.append(Failure(id=gid, kind="DIVIDES_VIOLATION", got=got, detail=f"degree {bad_div} does not divide |G|={t['order']}"))
                violated_here = True

            if isinstance(t.get("nr_conj_classes"), int) and len(got) != t["nr_conj_classes"]:
                failures.append(
                    Failure(
                        id=gid,
                        kind="LEN_VS_CONJ_CLASSES",
                        got=got,
                        detail=f"len={len(got)} != nr_conj_classes={t['nr_conj_classes']}",
                    )
                )
                violated_here = True

            if isinstance(t.get("abelianization_order"), int):
                c1 = count_eq(got, 1)
                if c1 != t["abelianization_order"]:
                    failures.append(
                        Failure(
                            id=gid,
                            kind="COUNT1_VS_ABELIANIZATION",
                            got=got,
                            detail=f"count(deg=1)={c1} != abelianization_order={t['abelianization_order']}",
                        )
                    )
                    violated_here = True

            solver_distinct_sets_all.append(distinct_sorted(got))

        if violated_here:
            groups_with_violations += 1

    return {
        "pass": len(failures) == 0,
        "n": targets_obj["n"],
        "style": solver_style,
        "summary": {
            "targets": len(targets_obj.get("targets", [])),
            "answered": len(answers),
            "missing": sum(1 for f in failures if f.kind == "MISSING_GROUPS"),
            "extra": sum(1 for f in failures if f.kind == "EXTRA_GROUPS"),
            "duplicate_ids": len(dups),
            "groups_with_violations": groups_with_violations,
        },
        "failures": [failure_to_dict(f) for f in failures],
        "derived": {"solver_distinct_degree_sets": unique_degree_sets(solver_distinct_sets_all)},
    }


def run_oracle_comparison(
    targets_obj: dict[str, Any],
    solver_obj: dict[str, Any],
    oracle_obj: dict[str, Any],
    solver_style: SolverStyle,
    spec_lock: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if spec_lock is None:
        spec_lock = build_spec_lock(int(targets_obj["n"]), solver_style)

    failures: list[Failure] = []

    if targets_obj.get("n") != solver_obj.get("n") or targets_obj.get("n") != oracle_obj.get("n"):
        failures.append(
            Failure(
                kind="N_MISMATCH",
                detail=(
                    f"targets.n={targets_obj.get('n')}, solver.n={solver_obj.get('n')}, "
                    f"oracle.n={oracle_obj.get('n')}"
                ),
            )
        )

    _check_spec_lock_or_fail(failures, solver_obj, spec_lock)

    oracle_by_id: dict[str, list[int]] = {}
    for o in oracle_obj.get("oracle", []):
        if not isinstance(o, dict):
            continue
        try:
            key = key_of_id(o.get("id"))
        except Exception:
            continue
        degrees = o.get("degrees")
        if is_pos_int_array(degrees):
            oracle_by_id[key] = degrees

    oracle_distinct_all: list[list[int]] = []
    for t in targets_obj.get("targets", []):
        exp = oracle_by_id.get(key_of_id(t["id"]))
        if exp:
            oracle_distinct_all.append(distinct_sorted(exp))
    oracle_distinct_sets = unique_degree_sets(oracle_distinct_all)

    solver_distinct_all: list[list[int]] = []
    exact_multiset_matches = 0
    exact_distinct_matches = 0
    strict_by_id_pass = False

    if solver_style == "GLOBAL_SET_OF_SETS":
        raw_sets = solver_obj.get("distinct_degree_sets")
        if not isinstance(raw_sets, list):
            failures.append(Failure(kind="PARSE_ERROR", detail="distinct_degree_sets must be an array"))
        else:
            for raw in raw_sets:
                if not is_pos_int_array(raw):
                    failures.append(Failure(kind="NON_POSINT_DEGREES", detail="each distinct_degree_set must be positive ints"))
                else:
                    solver_distinct_all.append(distinct_sorted(raw))
        strict_by_id_pass = False
    else:
        answers = solver_obj.get("answers")
        if not isinstance(answers, list):
            failures.append(Failure(kind="PARSE_ERROR", detail="answers must be an array"))
            answers = []

        dups = find_duplicate_ids(answers)
        if dups:
            failures.append(Failure(kind="DUPLICATE_IDS", detail=f"duplicate ids: {fmt([list(x) for x in dups])}"))

        answered_by_id: dict[str, Any] = {}
        target_set = {key_of_id(t["id"]) for t in targets_obj.get("targets", [])}

        for a in answers:
            if not isinstance(a, dict):
                continue
            try:
                answered_by_id[key_of_id(a.get("id"))] = a.get("degrees")
            except Exception:
                failures.append(Failure(kind="PARSE_ERROR", detail=f"invalid answer id format: {a.get('id')!r}"))

        for t in targets_obj.get("targets", []):
            gid = id_to_tuple(t["id"])
            k = key_of_id(gid)
            if k not in answered_by_id:
                failures.append(Failure(id=gid, kind="MISSING_GROUPS"))

        for a in answers:
            if not isinstance(a, dict) or "id" not in a:
                continue
            try:
                gid = id_to_tuple(a["id"])
                k = key_of_id(gid)
            except Exception:
                continue
            if k not in target_set:
                failures.append(Failure(id=gid, kind="EXTRA_GROUPS", detail="answer id not in target set"))

        for t in targets_obj.get("targets", []):
            gid = id_to_tuple(t["id"])
            k = key_of_id(gid)
            exp = oracle_by_id.get(k)
            got = answered_by_id.get(k)
            if exp is None or got is None:
                continue

            if not is_pos_int_array(got):
                failures.append(Failure(id=gid, kind="NON_POSINT_DEGREES", expected=exp, got=got, detail="degrees must be positive integers"))
                continue

            if 1 not in got:
                failures.append(Failure(id=gid, kind="MISSING_DEGREE_1", expected=exp, got=got))
                continue

            s = sumsq(got)
            if s != t["order"]:
                failures.append(Failure(id=gid, kind="SUMSQ_VIOLATION", expected=exp, got=got, detail=f"sum(d^2)={s} != |G|={t['order']}"))
                continue

            bad_div = next((d for d in got if t["order"] % d != 0), None)
            if bad_div is not None:
                failures.append(Failure(id=gid, kind="DIVIDES_VIOLATION", expected=exp, got=got, detail=f"degree {bad_div} does not divide |G|"))
                continue

            exp_distinct = distinct_sorted(exp)
            got_distinct = distinct_sorted(got)
            solver_distinct_all.append(got_distinct)

            if multiset_eq(exp, got):
                exact_multiset_matches += 1
            else:
                failures.append(Failure(id=gid, kind="WRONG_DEGREES", expected=exp, got=got))

            if sig_of_set(exp_distinct) == sig_of_set(got_distinct):
                exact_distinct_matches += 1
            else:
                failures.append(Failure(id=gid, kind="WRONG_DISTINCT_SET", expected=exp_distinct, got=got_distinct))

        strict_by_id_pass = len(failures) == 0

    solver_distinct_sets = unique_degree_sets(solver_distinct_all)
    diff = diff_degree_set_lists(oracle_distinct_sets, solver_distinct_sets)

    parse_like_kinds = {"PARSE_ERROR", "NON_POSINT_DEGREES", "SPEC_LOCK_MISMATCH", "N_MISMATCH"}
    parse_like_failures = [f for f in failures if f.kind in parse_like_kinds]
    global_pass = len(diff["missing"]) == 0 and len(diff["extra"]) == 0 and len(parse_like_failures) == 0

    if not global_pass:
        failures.append(
            Failure(
                kind="WRONG_GLOBAL_SET_OF_SETS",
                detail=f"missing={len(diff['missing'])}, extra={len(diff['extra'])}",
            )
        )

    next_actions = (
        [
            "Benchmark met: global set-of-sets matches GAP.",
            "Now harvest a proof plan and start Lean formalization.",
            "Increase difficulty: set mode=ALL for more targets; remove desc hints; move toward Lean-checked proof steps.",
        ]
        if global_pass
        else [
            "Fix the solver output (do NOT look at oracle degrees).",
            "Run pre-check and use the repair prompt until constraints pass.",
            "Then rerun oracle comparison.",
        ]
    )

    answered_count = (
        len(solver_obj.get("answers", []))
        if solver_style == "BY_GROUP_ID"
        else len(solver_obj.get("distinct_degree_sets", []))
        if isinstance(solver_obj.get("distinct_degree_sets"), list)
        else 0
    )

    return {
        "strict_by_id_pass": strict_by_id_pass if solver_style == "BY_GROUP_ID" else False,
        "global_set_of_sets_pass": global_pass,
        "n": targets_obj["n"],
        "summary": {
            "targets": len(targets_obj.get("targets", [])),
            "oracle": len(oracle_obj.get("oracle", [])),
            "answered": answered_count,
            "exact_multiset_matches": exact_multiset_matches,
            "exact_distinct_set_matches": exact_distinct_matches,
            "oracle_unique_distinct_sets": len(oracle_distinct_sets),
            "solver_unique_distinct_sets": len(solver_distinct_sets),
        },
        "failures": [failure_to_dict(f) for f in failures],
        "derived": {
            "oracle_distinct_degree_sets": oracle_distinct_sets,
            "solver_distinct_degree_sets": solver_distinct_sets,
            "missing_distinct_sets": diff["missing"],
            "extra_distinct_sets": diff["extra"],
        },
        "next_actions": next_actions,
    }


def build_export_artifact(
    n: int,
    solver_style: SolverStyle,
    lean_unlock_policy: UnlockPolicy,
    spec_lock: dict[str, Any],
    targets: Optional[dict[str, Any]],
    solver: Optional[dict[str, Any]],
    oracle: Optional[dict[str, Any]],
    precheck: Optional[dict[str, Any]],
    oracle_report: Optional[dict[str, Any]],
) -> dict[str, Any]:
    benchmark_output = None
    if oracle_report is not None:
        benchmark_output = {
            "solver_distinct_degree_sets": oracle_report.get("derived", {}).get("solver_distinct_degree_sets", []),
            "oracle_distinct_degree_sets": oracle_report.get("derived", {}).get("oracle_distinct_degree_sets", []),
        }

    return {
        "n": n,
        "solver_style": solver_style,
        "lean_unlock_policy": lean_unlock_policy,
        "spec_lock": spec_lock,
        "targets": targets,
        "solver": solver,
        "oracle": oracle,
        "precheck": precheck,
        "oracle_report": oracle_report,
        "benchmark_output": benchmark_output,
    }


def _run_self_tests() -> None:
    assert multiset_eq([1, 2, 1], [2, 1, 1])
    assert not multiset_eq([1, 2], [1, 1, 2])
    assert sumsq([1, 1, 2]) == 6
    assert is_pos_int_array([1, 2, 3])
    assert not is_pos_int_array([1, 0, 2])
    assert sig_of_set([1, 2, 3]) == "1,2,3"
    assert deep_eq({"b": 1, "a": [2, 3]}, {"a": [2, 3], "b": 1})

    targets = {
        "n": 8,
        "nr_groups": 1,
        "targets": [
            {
                "id": [8, 1],
                "order": 8,
                "nr_conj_classes": 5,
                "abelianization_order": 4,
            }
        ],
    }
    spec_lock = build_spec_lock(8, "BY_GROUP_ID")
    solver = {
        "n": 8,
        "spec_lock": spec_lock,
        "answers": [{"id": [8, 1], "degrees": [1, 1, 1, 1, 2]}],
    }
    oracle = {"n": 8, "oracle": [{"id": [8, 1], "degrees": [1, 1, 1, 1, 2]}]}

    pre = run_precheck(targets, solver, "BY_GROUP_ID", spec_lock)
    assert pre["pass"] is True

    cmp_report = run_oracle_comparison(targets, solver, oracle, "BY_GROUP_ID", spec_lock)
    assert cmp_report["global_set_of_sets_pass"] is True
    assert cmp_report["strict_by_id_pass"] is True


if __name__ == "__main__":
    _run_self_tests()
    print("agent_architecture_backend self-tests: OK")
