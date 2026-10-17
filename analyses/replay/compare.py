"""Compare two replay.py outputs: map the renames of the 2026-10-16 naming change (old
edge_fo / core_fo / step_fo / rim_fo against segment_fo / edge1_fo, old scan parameter
names against a_fo ...), then list every other difference. New summary columns show as
"only in new". Hand-check a few mapped values before trusting a 0.

usage: python analyses/replay/compare.py OLD.json NEW.json
"""
import json, re, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])

def num(v):
    return None if v in (None, "None", "nan") else float(v)

def norm_summary(row):
    row = dict(row)
    if "edge_fo" in row:  # old names
        vals = [row.pop("edge_fo")]
        step = row.pop("step_fo", None)
        if num(step) is not None:
            vals.append(step)
        vals.append(row.pop("core_fo"))
        row["segment_fo"] = [float(num(v)) for v in vals]
        row["edge1_fo"] = row.pop("rim_fo", None)
    else:
        row["segment_fo"] = [float(x) for x in row["segment_fo"].split(",")]
    if num(row.get("edge1_fo")) is None:
        row["edge1_fo"] = None
    for k in ("scan_best", "scan_sensitivity"):
        if isinstance(row.get(k), str):
            row[k] = rename_params(row[k])
    return row

def rename_params(text, n_seg=None):
    # old: edge_fo -> a_fo, core_fo -> last segment, step_fo -> b_fo (3 segments)
    if "core_fo" in text or "edge_fo" in text:
        last = "c_fo" if "step_fo" in text else "b_fo"
        text = text.replace("edge_fo", "a_fo").replace("step_fo", "b_fo").replace("core_fo", last)
    return text

def norm(key, v):
    if isinstance(v, dict):
        out = {k: norm(k, x) for k, x in v.items()}
        if "summary" in out:
            out["summary"] = norm_summary(out["summary"])
        for k in ("scan_best", "scan_sensitivity"):
            if isinstance(out.get(k), str):
                out[k] = rename_params(out[k])
        if "best_initial" in out:
            out["best_initial"] = numbers(out["best_initial"])
        return out
    if key.startswith(("paper_run:", "rr_run:")):
        return numbers(rename_params(v))
    return v

def numbers(text):
    """The numbers in a repr (old and new reprs order fields differently: sort them)."""
    return sorted(float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", text))

out = []
def walk(x, y, path):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                out.append(f"{path}/{k}: only in {'new' if k not in x else 'old'}")
            else:
                walk(x[k], y[k], f"{path}/{k}")
    elif x != y:
        out.append(f"{path}: {str(x)[:150]} -> {str(y)[:150]}")
A = {k: norm(k, v) for k, v in a.items()}
B = {k: norm(k, v) for k, v in b.items()}
walk(A, B, "")
print(len(out), "differences after mapping the renames")
for line in out[:60]:
    print(" ", line)
