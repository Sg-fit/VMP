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

**Oct 2026: plain and professional.** The first look (Futura, signal orange, uppercase labels, square black rules)
read as a fashion or gadget site. It's now a quiet working tool: the reader's own system font, greys, one calm accent,
light rules, small rounded corners. The rules below are the current ones.

**Two looks.** Professional is the default. The first design lives on as **Studio** (Futura, signal orange,
uppercase tracked labels with an accent square, heavy ink rules, square corners), switched in the footer
("Look: Professional / Studio"). It's one block of `html[data-look="studio"]` overrides at the end of
`templates/base.html`'s head, so pages are styled once and Studio only restyles them. The choice is kept in the
browser (localStorage) and applied before the page draws, so there's no flash; it never reaches the server.

- **Type:** the system UI font (`system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial`): what the
  reader's own apps use, and nothing loaded from a third party. Headings semibold (600), never heavier; headline at most
  2.3 rem. Sentence case everywhere: no uppercase or tracked-out labels. Monospace only for the drum/bass grids.
- **Colour:** light `#f7f7f8` page, white surfaces, ink `#1f2328`; dark `#16181b` / `#1d2024` / `#e6e8eb`. Secondary
  text and lines are the ink at reduced opacity. One accent, a calm navy `#1f3a5f` (dark mode `#9db8de`): the main
  button, links, the selected tab or option, focus. It's kept apart from the chart blue so a chart never looks like a
  button.
- **Charts** (`charts.py`, inline SVG): your own track is always drawn in the ink colour; other series use blue
  `#2a78d6`, orange `#eb6834`, aqua `#1baf7a` (dark mode `#3987e5`, `#d95926`, `#199e70`), in that fixed order. That
  set passes a colour-blind separation check against both surfaces (worst pair ΔE 9.2 light, 9.4 dark); aqua is
  under 3:1 on white, so charts always carry labels or sit next to the numbers. Thin marks, hairline axes, 4 px
  rounded bar ends, a 2 px surface gap between touching blocks, a 2 px surface ring on dots, `<title>` tooltips on
  every mark, a legend whenever there are two or more series. Labels never wear a series colour.
- **Shape:** 6 px corners on buttons and inputs, 8 px on panels, round chips; 1 px grey rules only (no heavy black
  rules, no decorative squares).
- **Logo:** five level-meter bars in the accent, drawn in CSS.
- **Motion:** the "analysing" spinner is a level meter that bounces; it stops for people who prefer reduced motion.
- **Emoji** are kept out of the interface (no padlocks or headphones on buttons) and stripped from report headings when shown on the page (the downloaded `.md`
  report is unchanged).
- **Density** (Oct 2026 redesign): pictures before paragraphs, and no big empty bands. Page padding 28 px, gaps
  between blocks 12-36 px, headline at most 2.3 rem. Every report section shows its gist and folds the rest into a
  **More** box; one sentence of help instead of a paragraph, longer help behind a tooltip or a fold.
- **Interactive by default:** every chart mark has a tooltip (hover or tap); charts on a track's timeline light up
  the same moment together and play from it in **Listen along** (the musician's own file, never uploaded); drum
  grids can be heard (sounds made in the browser); the home page's preview flips through four real report charts;
  chosen files show their waveform before anything is uploaded.

# Voice and tone

How the app talks. Based on reading (Oct 2026) the **Mailchimp Content Style Guide** (voice and tone), **Ableton's
Learning Music** lessons and **Bandcamp's** about page: three companies whose writing people trust.

What they share:

- **Plainspoken.** Mailchimp strips out hype and "fluffy metaphors"; clarity beats entertainment. Bandcamp says
  what happens to your money in one sentence, with the number.
- **Talk to one person.** Ableton: "you'll learn", "check out the boxes below", "next, you'll make some of your
  own". Short sentences in the second person that say what to do and what happens next.
- **Active and positive.** Say what something *does*, not what it isn't. Active voice ("we listen to the songs",
  not "songs are analysed").
- **Translate the expert words.** Explain a term the first time it shows up, in everyday words, without talking
  down.
- **Humour only when it comes naturally.** Dry, never forced. When in doubt, keep a straight face.
- **Tone follows the reader's mood.** Calm and short when something failed; a little warmer when they've finished.

How the app applies it:

1. **"We" for the app, "you" for the reader.** "We couldn't read this file", not "File unreadable".
2. **Contractions** (we'll, you're, isn't) and everyday words: "about", not "approximately"; "use", not "utilise".
3. **Lead with what they get, then the how.** Caveats go after the result, in one plain sentence, not a list of
   disclaimers.
4. **Honest about limits, without apologising at length.** "This is a guess from the sound, so check it by ear."
5. **Errors say what happened and what to do next**, in that order, with no blame.
6. **No exclamation marks in the interface**, no "simply" or "just" (it makes hard things sound easy), no
   marketing superlatives.
7. **Numbers stay concrete.** "About 30 seconds a song", "counts 0.4 against 1".
8. **Say it like a musician would.** "Songs you love", "the tune", "It's vocals, bass and guitar" - not "reference
   tracks", "melodic content", "instrumentation detected". Each track opens with a few plain sentences (*In short*)
   built only from what the report shows below.
