"""Dashboard views — one module per tab."""

from holodeck.dashboard.views.compare import render_compare
from holodeck.dashboard.views.explorer import render_explorer
from holodeck.dashboard.views.summary import render_summary

__all__ = ["render_summary", "render_explorer", "render_compare"]
