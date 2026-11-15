"""Design tokens for the dashboard: one neutral palette, one accent, two semantic colours.

Every colour used by the CSS (``ui.py``) and the charts (``charts.py``) comes from
here, so the look can be changed in one place. ``.streamlit/config.toml`` repeats
the base values for Streamlit's own widgets; keep the two in sync.

Rules
- Neutrals carry almost everything: background, text hierarchy, borders, context series.
- ``accent`` (a muted ink blue) marks the key series and primary actions - use sparingly.
- ``red`` means failure / alert only. ``green`` means "OK" only, and only where needed.
"""

TOKENS = {
    "light": {
        "bg": "#ffffff",
        "sidebar": "#fafafa",
        "subtle": "#f4f4f5",       # quiet fills (inputs, neutral heatmap cells)
        "text": "#18181b",
        "text2": "#52525b",        # secondary text
        "muted": "#71717a",        # labels, captions, axis text
        "border": "#e4e4e7",
        "grid": "rgba(24,24,27,0.07)",
        "accent": "#2d5b88",
        "accent_soft": "rgba(45,91,136,0.10)",
        "on_accent": "#ffffff",
        "red": "#b23b3b",
        "red_soft": "rgba(178,59,59,0.12)",
        "green": "#3d7a57",
        # Context series, darkest first (most important non-key series gets the darkest).
        "greys": ["#5f5f68", "#9a9aa2", "#c8c8ce"],
        # Sequential (probability) and diverging (correlation) scales.
        "seq": [[0, "#f4f4f5"], [0.5, "#e3b5b0"], [1, "#a33a3a"]],
        "div": [[0, "#2d5b88"], [0.5, "#f4f4f5"], [1, "#b23b3b"]],
        "cell": "#f4f4f5", "cell_warn": "#f7e6e4", "cell_bad": "#f0d2ce",
    },
    "dark": {
        "bg": "#111113",
        "sidebar": "#141416",
        "subtle": "#1c1c1f",
        "text": "#ededef",
        "text2": "#b4b4bb",
        "muted": "#8b8b93",
        "border": "#2a2a2f",
        "grid": "rgba(255,255,255,0.07)",
        "accent": "#7da3cc",
        "accent_soft": "rgba(125,163,204,0.14)",
        "on_accent": "#111113",
        "red": "#d46a6a",
        "red_soft": "rgba(212,106,106,0.16)",
        "green": "#6dae88",
        "greys": ["#a1a1aa", "#71717a", "#4a4a51"],
        "seq": [[0, "#1c1c1f"], [0.5, "#6b3a3a"], [1, "#e07a7a"]],
        "div": [[0, "#7da3cc"], [0.5, "#1c1c1f"], [1, "#d46a6a"]],
        "cell": "#1c1c1f", "cell_warn": "#2e2122", "cell_bad": "#3d2526",
    },
}

FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
DASHES = ["dash", "dot", "dashdot"]  # non-key series also differ by line style, not colour alone


def tokens(mode):
    return TOKENS["light" if mode == "light" else "dark"]


def status_color(tier, mode):
    """Health-state colours: red for at-risk, a neutral grey for watch, muted green for healthy."""
    t = tokens(mode)
    return {"critical": t["red"], "watch": t["greys"][0], "healthy": t["green"], "unknown": t["greys"][1]}[tier]
