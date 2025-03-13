from marimo._ast.app import InternalApp
from __init__ import app
from pathlib import Path

# Lets move out the code first

order = InternalApp(app).execution_order
codes = {k: v.code for k, v in InternalApp(app).graph.cells.items() if v.language=="python" and "## Export" in v.code}

code_export = ""
for i in order:
    if i in codes:
        code_export += codes[i].replace("## Export", "") + "\n"

Path("simsity/__init__.py").write_text(code_export)
