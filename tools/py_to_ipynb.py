"""Convert a 'percent-format' .py script (cells separated by `# %%`) into a Jupyter notebook.

Usage:  python tools/py_to_ipynb.py part1_telco_churn/churn_crispdm.py
Writes the .ipynb next to the script. No third-party dependencies.
"""
import json
import sys
import uuid
from pathlib import Path


def convert(py_path: Path) -> Path:
    lines = py_path.read_text(encoding="utf-8").splitlines()
    cells, kind, buf = [], "code", []

    def flush():
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "markdown":
            md = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in text.splitlines()]
            cells.append({"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": "\n".join(md)})
        else:
            cells.append({"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {}, "execution_count": None,
                          "outputs": [], "source": text})

    for ln in lines:
        if ln.startswith("# %%"):
            flush()
            buf = []
            kind = "markdown" if "[markdown]" in ln else "code"
            continue
        buf.append(ln)
    flush()

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = py_path.with_suffix(".ipynb")
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("wrote", convert(Path(arg)))
