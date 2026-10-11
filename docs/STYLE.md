# Visual style

How the web app looks, and why. Based on a study (Oct 2026) of three music sites made by human designers:
**Ableton**, **Bandcamp** and **Hooktheory**, measured from their live pages (fonts, colours, sizes, corners).

## What the three sites have in common

| | Ableton | Bandcamp | Hooktheory |
|---|---|---|---|
| Typeface | Futura (geometric), one family for everything | Helvetica Neue, one family | Gotham (geometric), one family |
| Colours | black on light grey `#f3f3f3`, white; **one** accent (pure blue) + small flat colour tags | charcoal `#222` with off-white text `#ecf3f4`; almost no colour | black; **one** accent (blue) |
| Hierarchy | size and weight | size, weight and **opacity** (white at 72% for secondary text, 8% for lines) | huge headline (70 px, tight letter-spacing), small uppercase label above it |
| Corners | square (0 px) everywhere | square (0 px) on ~95% of elements, 4 px on small controls | pill buttons, rounded cards |
| Decoration | none: flat colour blocks and photography | hand-drawn sparkles, real album art | product screenshots |
| Numbers | — | live sales figures as content | big stats ("1.5M+ projects") |

The rules that follow from that:

1. **One typeface, used with confidence.** Big, bold, tightly tracked headlines; plain body text. No mixing.
2. **Monochrome plus one accent.** The accent marks the one thing to do (the main button, focus). Everything else is
   black, white and greys made with opacity.
3. **Hierarchy from type, not boxes.** Fewer cards and borders; hairline rules and whitespace separate things.
4. **Square, flat, no shadows.** Corners are square; nothing floats.
5. **Labels are small and uppercase**, with letter-spacing (the "eyebrow" above a headline, table headers).
6. **Numbers are display elements.** Tempo, key and counts are shown large, like a hardware readout.
7. **No emoji in the interface.** These sites use typography, icons or real imagery; emoji read as unfinished.

## How the app applies it

- **Type:** a geometric sans stack (`Futura, "Avenir Next", Avenir, "Century Gothic", system-ui`), so it uses what is
  installed and never loads fonts from a third party. Headlines 700 weight, -0.02em tracking, tight line height.
  Monospace only for the drum/bass grids.
- **Colour:** light `#f3f3f1` / ink `#111`; dark `#191919` / `#ecf0f1`. Secondary text and lines are the ink colour at
  reduced opacity. Accent: signal orange `#ff5a1f` (a hardware-LED colour, our own rather than borrowed), always with
  black text on it for contrast.
- **Shape:** 0 px corners on cards, buttons and inputs; 2 px on inline code. 1 px hairlines; a 2 px ink rule above
  each track in the report.
- **Logo:** five level-meter bars, drawn in CSS.
- **Motion:** the "analysing" spinner is a level meter that bounces; it stops for people who prefer reduced motion.
- **Emoji** are stripped from the interface and from report headings when shown on the page (the downloaded `.md`
  report is unchanged).
