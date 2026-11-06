"""Small line icons for the pages: one stroke weight, drawn in the text's own colour (so they follow light, dark and
the Studio look), and hidden from screen readers because the words next to them already say it."""
from markupsafe import Markup

# 24 x 24, stroke only
PATHS = {
    "lock": "M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3",
    "shield": "M12 3l7 3v5c0 5-3.2 8.4-7 10-3.8-1.6-7-5-7-10V6zM9 12l2 2 4-4",
    "user": "M12 4.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zM5 20a7 7 0 0 1 14 0",
    "users": "M9 6a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM3 19a6 6 0 0 1 12 0M16.5 7.5a2.5 2.5 0 1 1 0 5M17 14.5a5 5 0 0 1 4 4.5",
    "wave": "M3 12h1.5M7 8.5v7M10.5 5v14M14 8v8M17.5 10v4M21 12h-1",
    "spark": "M12 3v5M12 16v5M3 12h5M16 12h5M6.3 6.3l2.5 2.5M15.2 15.2l2.5 2.5M6.3 17.7l2.5-2.5M15.2 8.8l2.5-2.5",
    "gauge": "M4 16a8 8 0 1 1 16 0M12 16l4-5M7 16h.01M17 16h.01",
    "info": "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 11v5M12 8h.01",
    "chart": "M4 20V11M10 20V5M16 20v-7M2 20h20",
    "compare": "M4 5h6v14H4zM14 9h6v10h-6z",
    "folder": "M3 6h6l2 2h10v11H3z",
    "bookmark": "M6 3h12v18l-6-4-6 4z",
    "server": "M4 4h16v6H4zM4 14h16v6H4zM8 7h.01M8 17h.01",
    "clock": "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 8v4l3 2",
    "music": "M9 18V6l11-2v12M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0zM20 16a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z",
    "drum": "M4 8c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 8v8c0 1.7 3.6 3 8 3s8-1.3 8-3V8M8 10.6v8M16 10.6v8",
}


def icon(name, size=20, cls="ico"):
    """The named icon as inline SVG; an unknown name draws nothing rather than a broken image."""
    d = PATHS.get(name)
    if not d:
        return Markup("")
    return Markup(f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                  f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
                  f'aria-hidden="true" focusable="false"><path d="{d}"/></svg>')
