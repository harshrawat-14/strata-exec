"""
Report module-level names that are referenced but never defined or imported.

Used after mechanical refactors (moving code between modules) to catch the
NameErrors that only appear at runtime on untested branches.

    python scripts/check_undefined_names.py rl/evaluation/*.py
"""
import builtins
import symtable
import sys


def check(path: str) -> int:
    top = symtable.symtable(open(path).read(), path, "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    bad = 0

    def walk(table):
        nonlocal bad
        for s in table.get_symbols():
            module_ref = table.get_type() == "module" and s.is_referenced()
            if (s.is_global() or module_ref) and s.is_referenced():
                name = s.get_name()
                if name not in defined and not hasattr(builtins, name) and name not in ("__file__", "__name__"):
                    print(f"{path}: undefined name {name!r} in {table.get_name()}")
                    bad += 1
        for child in table.get_children():
            walk(child)

    walk(top)
    return bad


if __name__ == "__main__":
    total = sum(check(p) for p in sys.argv[1:])
    print(f"undefined names: {total}")
    sys.exit(1 if total else 0)
