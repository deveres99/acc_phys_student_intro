"""
Execute notebooks end to end in a fresh Jupyter kernel and report problems.

    python tools/run_transverse_notebooks.py transverse_beam_dynamics/solutions/*.ipynb
    python tools/run_transverse_notebooks.py --student transverse_beam_dynamics/student/*.ipynb
    python tools/run_transverse_notebooks.py --save-dir out/ transverse_beam_dynamics/solutions/ex05*.ipynb

* Solutions mode (default): every cell must run without error.
* ``--student``: the first error must be a ``NotImplementedError`` (an
  unfinished stub), or there must be no error at all (notebooks without code
  stubs).  Anything else (SyntaxError, NameError before any stub...) is a failure.
* ``--save-dir DIR``: write the executed notebooks (with outputs) and every
  PNG figure to DIR, so you can look at them.

Needs only ``jupyter_client`` and ``ipykernel`` (no nbconvert/nbclient).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from queue import Empty

from jupyter_client.manager import start_new_kernel


def run_notebook(path: Path, student: bool, timeout: float, save_dir: Path | None):
    nb = json.loads(path.read_text(encoding="utf-8"))
    km, kc = start_new_kernel(kernel_name="python3", cwd=str(path.parent))
    # headless plotting
    kc.execute_interactive("import matplotlib; matplotlib.use('module://matplotlib_inline.backend_inline')",
                           timeout=60, output_hook=lambda m: None)
    t0 = time.time()
    first_error, n_images, slow = None, 0, []
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            outputs = []

            def hook(msg, outputs=outputs):
                t, c = msg["msg_type"], msg["content"]
                if t == "stream":
                    outputs.append({"output_type": "stream", "name": c["name"], "text": c["text"]})
                elif t in ("display_data", "execute_result"):
                    o = {"output_type": t, "data": c["data"], "metadata": c.get("metadata", {})}
                    if t == "execute_result":
                        o["execution_count"] = c.get("execution_count")
                    outputs.append(o)
                elif t == "error":
                    outputs.append({"output_type": "error", "ename": c["ename"],
                                    "evalue": c["evalue"], "traceback": c["traceback"]})

            tc = time.time()
            try:
                reply = kc.execute_interactive(src, timeout=timeout, output_hook=hook)
            except (TimeoutError, Empty):
                first_error = first_error or (i, "Timeout", f"cell took > {timeout} s")
                break
            dt = time.time() - tc
            if dt > 10:
                slow.append((i, round(dt, 1)))
            cell["outputs"] = outputs
            cell["execution_count"] = reply["content"].get("execution_count")
            for o in outputs:
                if "data" in o and "image/png" in o["data"]:
                    n_images += 1
                    if save_dir is not None:
                        img = save_dir / f"{path.stem}_cell{i:03d}_{n_images}.png"
                        img.write_bytes(base64.b64decode(o["data"]["image/png"]))
            if reply["content"]["status"] == "error":
                c = reply["content"]
                first_error = (i, c["ename"], c["evalue"])
                if not student:
                    tb = "\n".join(c["traceback"])
                    # strip ANSI colour codes for readability
                    import re
                    first_error = (i, c["ename"], c["evalue"] + "\n" + re.sub(r"\x1b\[[0-9;]*m", "", tb)[-3000:])
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    elapsed = time.time() - t0
    if save_dir is not None:
        (save_dir / path.name).write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return first_error, elapsed, n_images, slow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="+")
    ap.add_argument("--student", action="store_true")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
    failures = 0
    for nbp in args.notebooks:
        path = Path(nbp).resolve()
        err, dt, nimg, slow = run_notebook(path, args.student, args.timeout, save_dir)
        if args.student:
            # fine: stops at an unfinished stub, or has no code stubs at all and runs through
            ok = err is None or err[1] == "NotImplementedError"
            status = "OK  " if ok else "FAIL"
            detail = f"stops at cell {err[0]} with {err[1]}" if err else "runs to the end (no code stubs)"
        else:
            ok = err is None
            status = "OK  " if ok else "FAIL"
            detail = "" if ok else f"cell {err[0]}: {err[1]}: {err[2]}"
        failures += not ok
        slow_txt = f"  slow cells: {slow}" if slow else ""
        print(f"{status} {path.name:45s} {dt:6.1f} s  {nimg:3d} figs  {detail}{slow_txt}", flush=True)
    print(f"\n{len(args.notebooks) - failures}/{len(args.notebooks)} notebooks OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
