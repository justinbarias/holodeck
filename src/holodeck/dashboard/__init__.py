"""Dash dashboard for the eval-runs viewer.

Imported ONLY when the `dashboard` extra is installed. Nothing outside this
package may import from here. The CLI guards entry via `find_spec("dash")`
before spawning the dashboard subprocess.

This package is distinct from `holodeck.lib.ui/`, which contains terminal
rendering utilities (`colors.py`, `spinner.py`, `terminal.py`). The two
packages render different surfaces — Dash HTML/React vs. ANSI TTY — and
must not cross-import.
"""
