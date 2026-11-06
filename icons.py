"""Small line icons for the pages: one stroke weight, drawn in the text's own colour (so they follow light, dark and
the Studio look), and hidden from screen readers because the words next to them already say it."""
from markupsafe import Markup

# 24 x 24, stroke only
PATHS = {
    "trash": "M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3",
    "lock": "M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 0 1 7 0v3",
    "shield": "M12 3l7 3v5c0 5-3.2 8.4-7 10-3.8-1.6-7-5-7-10V6zM9 12l2 2 4-4",
    "ban": "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM6.4 6.4l11.2 11.2",
    "user": "M12 4.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7zM5 20a7 7 0 0 1 14 0",
    "users": "M9 6a3 3 0 1 0 0 6 3 3 0 0 0 0-6zM3 19a6 6 0 0 1 12 0M16.5 7.5a2.5 2.5 0 1 1 0 5M17 14.5a5 5 0 0 1 4 4.5",
    "disc": "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 9.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5z",
    "wave": "M3 12h1.5M7 8.5v7M10.5 5v14M14 8v8M17.5 10v4M21 12h-1",
    "spark": "M12 3v5M12 16v5M3 12h5M16 12h5M6.3 6.3l2.5 2.5M15.2 15.2l2.5 2.5M6.3 17.7l2.5-2.5M15.2 8.8l2.5-2.5",
    "trend": "M3 17l6-6 4 4 8-8M15 7h6v6",
    "phone": "M8 3h8a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1H8a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM11 18h2",
    "eyeoff": "M3 3l18 18M10.6 6.1A9.5 9.5 0 0 1 12 6c5 0 9 6 9 6a15.6 15.6 0 0 1-3 3.4M6.6 7.6C4.3 9.2 3 12 3 12s4 6 9 6"
             "a8.6 8.6 0 0 0 4-1",
    "compass": "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM15 9l-1.8 4.2L9 15l1.8-4.2z",
    "gauge": "M4 16a8 8 0 1 1 16 0M12 16l4-5M7 16h.01M17 16h.01",
    "info": "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 11v5M12 8h.01",
    "chart": "M4 20V11M10 20V5M16 20v-7M2 20h20",
    "compare": "M4 5h6v14H4zM14 9h6v10h-6z",
    "folder": "M3 6h6l2 2h10v11H3z",
    "bookmark": "M6 3h12v18l-6-4-6 4z",
    "link": "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    "download": "M12 4v11M7 10l5 5 5-5M4 20h16",
    "server": "M4 4h16v6H4zM4 14h16v6H4zM8 7h.01M8 17h.01",
    "clock": "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 8v4l3 2",
    "guitar": "M14 10l6-6M17 3l4 4M13.5 10.5a3 3 0 0 0-3.9-.4 4 4 0 0 0-5.4 5.6 4 4 0 0 0 5.6 3.6 3 3 0 0 0 3.7-3.1"
              "M8 14h.01",
    "music": "M9 18V6l11-2v12M9 18a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0zM20 16a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0z",
    "speaker": "M4 9h4l5-4v14l-5-4H4zM16 9a4 4 0 0 1 0 6M18.5 6.5a7.5 7.5 0 0 1 0 11",
    "drum": "M4 8c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 8v8c0 1.7 3.6 3 8 3s8-1.3 8-3V8M8 10.6v8M16 10.6v8",
    "metronome": "M9 3h6l3 18H6zM12 17l4-9M8.5 14h7",
    "layers": "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5",
    "blocks": "M3 6h5v12H3zM10 6h5v12h-5zM17 6h4v12h-4z",
    "check": "M5 12.5l4.5 4.5L19 7.5",
}

# The idea areas the analysis writes (musicanalyze: "**harmony:** ..."), and the icon each one's card carries.
AREA_ICONS = {"harmony": "music", "melody": "wave", "bass": "guitar", "drums": "drum", "rhythm": "metronome",
              "arrangement": "layers", "structure": "blocks", "sound": "speaker", "tuning": "gauge", "remix": "spark"}


def icon(name, size=20, cls="ico"):
    """The named icon as inline SVG; an unknown name draws nothing rather than a broken image."""
    d = PATHS.get(name)
    if not d:
        return Markup("")
    return Markup(f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
                  f'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
                  f'aria-hidden="true" focusable="false"><path d="{d}"/></svg>')
