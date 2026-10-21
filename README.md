# 🎵 Music Inspiration Analyzer

Analyzes instrumental tracks (no vocals needed) and turns the analysis into brainstorming material for musicians.

| Section | What you get |
|---|---|
| **Harmony** | Key and [Camelot](https://mixedinkey.com/camelot-wheel/) code, key timeline, time spent in each key, **mode** (Dorian, harmonic minor, Mixolydian…) with its signature "colour note", the notes used most, the chords used most, harmonic rhythm, **repeating chord loops** with Roman numerals (for example `Bm – Em – F#` = i – iv – V), and the full chord timeline |
| **Structure** | Sections (A, B, A…) with times, energy level, key and main chords. A section also changes where the drums stop or start for 8 seconds or more (a breakdown under the same chords), and each boundary sits on the sharpest change in level within 4 seconds, on a beat. Recognised forms: 8-, 12- and 16-bar blues (major and minor, with correct variations to try) and the 32-bar AABA song form. Meters: 3/4, 4/4, 5/4, 7/4 |
| **⚡ Energy & mood** | Measured on the audio, per section: energy 1–10 (loudness, notes per second, brightness, tempo) and mood/positivity (mode, minor-chord share, brightness, tempo), with words like *calm*, *driving* or *melancholic*, plus the track's energy shape (builds, peaks, winds down) |
| **🎯 Next to the songs you admire** (optional, first in each track's report) | Add up to three finished songs you admire, from any genre, and each of your tracks is measured beside them on one scale: tempo, key and mode, length and sections, how soon it reaches full energy, energy contrast, off-beat feel, drums, bass and drum pattern, how often the chords change, loudness, punch (peak-to-loudness ratio), loudness range, stereo width and a mono low end, brightness and frequency balance, and for dance tracks the first drop, how hard it lands and sidechain pumping. Only differences big enough to hear are pointed out (about 4% tempo, 1 LU loudness, 3 dB in a frequency band or in pumping, a third of an octave in brightness), each with something to try. It opens with **where you stand** (already close in…, different in…), then **what all of them share that yours doesn't** (a difference every song you admire has is a stronger lead than one song's quirk), the **energy curves and tonal balance laid over each other**, and each song side by side. References get no ideas, MIDI or sheet music of their own, aren't counted in your profile, and are deleted with your audio. Command line: `--reference a.mp3 --reference b.mp3` |
| **🎛️ Producer's view** (electronic and other produced tracks) | For tracks with drums played to a grid, or when you pick an electronic genre or tick synth: **sidechain pumping** (how far the pads duck after each kick, judged on the shape of the beat: fast duck, slow swell), the **dance arrangement** (intro, build, drop, breakdown, outro, each in bars, with how many dB the drop lands above its lead-in and whether sections fall on 8-bar phrases), a **mono low end** check below 120 Hz (club systems and phones play bass in mono), **punch** as peak-to-loudness ratio, and a DJ's **mixing range** (tempos within ~6% and Camelot neighbour keys). Ideas follow from the gaps: sidechain a four-on-the-floor track that doesn't pump, mono a wide low end, add a breakdown, make the drop land, give DJs a drums-only intro. Genres now include house, techno, trance, dubstep, UK garage, drum & bass, trap, synthwave and lo-fi. Rules and thresholds: `electronic.py` |
| **For guitar** | A capo position that turns the song's chords into open shapes, with fingerings (only suggested when it clearly helps; power chords left out) |
| **Instruments we hear** | Where the style model is installed (see below), a trained instrument model (Essentia, MTG-UPF: 40 instruments) lists which instruments it hears in each track and how sure it is, and points out a sure one you didn't tick. It's a second opinion: shown, and compared with the songs you admire ("theirs features saxophone; in yours it hardly registers"), never used to change a result. Drums and bass are compared from our own measurements. `MUSIC_HEAR=0` switches it off |
| **Instruments** | Drums separated from the rest: the main drum groove as a 16-step grid (kick/snare/cymbal; each drum voice counts equally, and the kick is judged by its power, so a house track's off-beat hats and bass can't drown it out), how repetitive it is, likely fills, and which sections have drums. Bass line: main notes, range, style (pedal, roots, riff, walking), how often it plays the chord root, and slash chords (A/E) in the chord timeline. The bass is transcribed note by note and written out as **sheet music** (bars of the detected meter, chord symbols above); its rhythm analysis (note lengths, off-beat and syncopated share, the most common bar rhythm) is read from that sheet |
| **Groove & sound** | Pulse steadiness, how busy the rhythm is, syncopation, percussive vs. tonal balance, tone and texture, energy arc, biggest build, quietest moment, frequency balance from sub to air. **Loudness as streaming services measure it**: integrated LUFS and true peak (ITU-R BS.1770), loudness range (EBU Tech 3342) and stereo correlation, plus what Spotify's published normalisation would do to the track (turned down to −14 LUFS, or lifted only as far as a −1 dBTP true peak allows) and warnings for clipping, hot peaks and phase problems |
| **About the music** (optional) | Tick the instruments that play (36 of them, grouped: drum kit, drum machine/808, hand percussion; bass guitar, double bass, synth bass; acoustic, electric and classical guitar, ukulele, banjo, mandolin; piano, electric piano, organ, accordion, mallets; synth pads, lead synth, samples; lead and backing vocals; violin, viola, cello, string section, harp; flute, clarinet, saxophone, trumpet, trombone, harmonica) and pick a genre when uploading. Each instrument belongs to a family the analysis reasons with, and brings its own range (a solo cello's low part reads from C2, a ukulele's from C4), its own sound in the MIDI files, guitar notation and capo shapes only for six-string guitars, and its name in the report. Your answers settle what audio alone can't: drums and a separate bass (a phone can't tell a bass guitar from a guitar's low strings), half/double-speed tempo readings ("you probably feel this at 174 BPM"). They also set the MIDI sound, write a guitar's low part in guitar notation, label a sung melody, add a genre-specific idea and give the AI context. Limit: apart from the drums (grid) and the lowest part (bass, or a solo instrument's own low notes), instruments aren't transcribed one by one; a guitar or piano in a band is covered by the chords and melody line. Command line: `--instruments classical_guitar,voice --genre blues` |
| **🎶 Melody** | Type the melody or riff you hear (e.g. `A E F E A E D E`) to get its scale degrees, which modes it fits, its shape, melodic devices, chords that harmonize each note, and generated variations (inversion, retrograde, sequence, answer phrase) as MIDI files. Each note is also **checked against the recording**: it flags notes that barely sound, and says whether the audio supports the order. There's also an automatic "top line" that says honestly when it's only following the chords |
| **🎧 Shared DNA** | Building blocks the track shares with well-known recordings: chord loops matched against famous progressions (for example i–IV = Santana's “Oye Como Va”), the mode, the drum groove, and melodic devices. The AI adds more reference tracks, labeled as suggestions to double-check |
| **💡 Ideas to try** | Rule-based suggestions that need no AI: how to use the mode's colour note, chords to borrow from the parallel key, smooth and dramatic key-change targets, half-time or double-time variants, breakdowns and builds, groove contrasts, gaps in the mix |
| **🤖 AI brainstorm** (optional) | A description of the feel, 6–8 specific ideas tied to times and chords, 2–3 alternative chord progressions, and title ideas |
| **🎼 Sheet music** | The bass part as MusicXML (opens in MuseScore, Sibelius, Dorico and most DAWs) and as an engraved page to view or print to PDF |
| **🎹 MIDI sketches** | `.mid` files of the bass (as played, and as written on the sheet), each detected chord loop, a variation of each loop, and the AI's progressions. Drag them into any DAW. |
| **🔗 Tracks that blend well** | With several files: which tracks are key-compatible (on the Camelot wheel) and close in tempo, for medleys, mashups or set lists |

It comes as a **web app** (upload tracks in the browser) and a **command-line tool**. Only the analysis numbers are sent to the AI, never the audio.

**Your memory** (signed-in accounts, off until switched on at *Profile → What we remember*): the app learns from
your saved analyses, the corrections you make on a result ("it's 3/4", "there's no bass") and the 👍/👎 you give
ideas, plus goals you write down. It uses them only to nudge: the upload form is pre-filled with your usual line-up
and genre; each new track gets **Next to your earlier tracks** (a first minor key, faster than anything you've
saved, chords you haven't used); close calls get a note (an uncertain tempo where half or double sits in your usual
range), never a changed result; ideas you disliked are dropped, liked kinds come first and ones you've seen go
last; the AI gets a short profile of your habits and goals. Everything is listed on that page, deletable one by one
or all at once ("Forget everything"); anonymous use is unaffected. Logic and rules: `memory.py`.

**Demo reports** (`/demo/blues`, `/demo/band`, `/demo/electronic`, linked from the home page) show a finished report
before anyone uploads: a solo guitar blues, a band song compared with a reference, and a house track compared with
two club tracks it admires. They're made
from generated audio (the self-test's synthesizers, so no rights issues), built the first time someone opens them
(the deploy self-test does that) and shared by everyone; they can't be saved into a private history.

## Find music like this (tags for a playlist)

A second mode, for listeners rather than musicians: describe a set of songs in the words music sites search by,
then search with those words to reach artists the big sites' recommendations rarely show.

- **Input:** upload songs, paste a public **Spotify** playlist/album/track link or a **YouTube** playlist link, or
  type song names (`Artist - Title`, one per line). Up to 100 songs.
- **Uploaded songs are listened to** (weight 1.0): a quicker "light" analysis (tempo, time signature, energy and
  mood, drums or not, blues forms) plus, when installed, a trained **style model** — Essentia's Discogs-EffNet
  (400 Discogs styles, moods/themes, instruments, vocals or not).
- **Songs from a link or a list are looked up by name** (weight 0.4) on MusicBrainz (and Last.fm when
  `LASTFM_API_KEY` is set). Those are listeners' tags, mostly for the artist, so they count for less. Only the names
  are read from a playlist page, never the audio.
- **Lesser-known artists with this sound:** artists carrying the list's most specific tags (MusicBrainz, searched
  in pairs) that far fewer people play than the list's own artists: between 20 listeners and a quarter of the
  list's typical artist (at most 10,000), counted by ListenBrainz, the open listening database (no key needed).
  Ranked by how many of the list's tags they share; artists already in the list are left out.
- **Describe what you're after** (optional): a few comma-separated words ("melancholic, trip hop") become tags of
  their own and lead the searches; works with no songs at all.
- **Output:** each tag's share of the list, broad genres kept apart (shown, never searched), and links to the tag
  pages on Bandcamp, Last.fm, MusicBrainz and RYM plus two-tag YouTube/Spotify searches. Artists already in
  the list are named so you can skip them.
- **Kept separate:** a tag job runs none of the musician's work (no AI, MIDI, sheet music, ideas), and a
  musician's analysis never runs the tag search.
- **Licence:** the Essentia models are **CC BY-NC-SA 4.0 — non-commercial use only** (Music Technology Group,
  Universitat Pompeu Fabra). If this app is ever used commercially, build without them (`WITH_STYLE_MODEL=0`)
  or license them from MTG.
- Command line: `python discover.py song.mp3 --songs list.txt --playlist URL --out tags/`

## Run the web app locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:8000, drop in your tracks, and you'll get the report on the page, plus a `.zip` with the report, JSON data and MIDI files.

## Deploy on your server

### Option A: Docker (recommended)

```bash
git clone https://github.com/<you>/<repo>.git
cd <repo>
cp .env.example .env        # then edit .env: API key
docker compose up -d --build
```

The app listens on **127.0.0.1:8090** (only reachable from the server itself; change `HOST_PORT` if 8090 is taken). Put your reverse proxy or Cloudflare Tunnel in front of it, see below. Job data lives in the `music-data` volume and is deleted automatically after `JOB_TTL_HOURS`.

### Option B: plain Python (Linux)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...
gunicorn --workers 1 --threads 4 --timeout 120 --bind 127.0.0.1:8090 app:app
```

Use **one** worker. Analysis runs in a background thread inside that process, so extra workers wouldn't see each other's jobs. To keep it running, wrap the command in a systemd service.

### Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | Turns on the AI brainstorm (optional) |
| `MUSIC_MODEL` | provider default | Use a different model |
| `UPLOADS_PER_HOUR` | 30 | How many analyses one visitor can start per hour (analysis is open to everyone) |
| `SECRET_KEY` | random, kept in the data volume | Signs the sign-in cookie |
| `MAX_UPLOAD_MB` | 95 | Max upload size per request (Cloudflare's free plan caps uploads at 100 MB) |
| `MAX_FILES` | 10 | Max tracks per upload |
| `JOB_TTL_HOURS` | 24 | How long results are kept |
| `DATA_DIR` | `./jobs` | Where results are stored |
| `HOST_PORT` | 8090 | Docker only: the localhost port your proxy forwards to |
| `MUSIC_SEPARATION` | `auto` | `auto` uses Demucs if installed, `demucs` requires it, `simple` always uses the fast built-in split |
| `MUSIC_MODEL` / `MUSIC_FALLBACK_MODELS` | provider default | Main AI model, and backups that are raced in if it's slow or fails. Run `--bench-llm` (below) to pick the fastest |
| `MUSIC_AI_HEDGE_SECONDS` / `MUSIC_AI_DEADLINE_SECONDS` | 30 / 150 | Ask a backup model in parallel after this many seconds; give up on AI ideas after this many |
| `MUSIC_LLM_BASE_URL` | NVIDIA's API | Any OpenAI-compatible endpoint, for example a self-hosted model |
| `MUSIC_CACHE_DIR` | `DATA_DIR/cache` (web), `~/.cache/music-analyzer` (CLI) | Cached analyses. A re-upload of the same recording skips the slow analysis. Entries unused for 30 days are deleted |
| `CPU_LIMIT` / `MEM_LIMIT` | 1.0 / 1500m | Docker only: caps on CPU cores and memory for the analyzer, so jobs can't slow down other sites on the server. A job that needs more memory fails with a message instead of freezing the machine |
| `WITH_STYLE_MODEL` | 1 | Docker build only: installs the Essentia style model for "Find music like this" in its own Python environment (Linux x86-64; a failed install only turns style tags off). `0` = smaller image |
| `LASTFM_API_KEY` | — | Optional: also look songs up on Last.fm (richer listener tags than MusicBrainz) |
| `MUSIC_OFFLINE` | — | `1` turns off looking songs up by name and reading playlist links |
| `WITH_DEMUCS` | 0 | Docker build only: `1` installs Demucs AI drum separation (about 1 GB larger image, 1–2 min of CPU per track). Rebuild with `docker compose up -d --build` after changing it |

### Behind a domain (nginx)

Put it behind nginx or Caddy for HTTPS. With nginx, raise the upload limit and the timeout:

```nginx
location / {
    proxy_pass http://127.0.0.1:8090;
    proxy_set_header Host $host;
    client_max_body_size 100m;
    proxy_read_timeout 120s;
}
```

### With Cloudflare (server already hosting other sites)

1. **DNS:** in Cloudflare, add an `A` record for a subdomain (for example `music`) pointing to your server's IP, with the orange cloud **Proxied** on.
2. **Reverse proxy:** add a site for `music.yourdomain.com` to your existing nginx or Caddy that forwards to `http://127.0.0.1:8090`, as in the nginx example above.
3. **SSL/TLS mode:** use **Full (strict)**, with a Cloudflare Origin Certificate or a Let's Encrypt certificate on the server.

If you use a **Cloudflare Tunnel** instead, add a public hostname `music.yourdomain.com` → `http://localhost:8090` to your existing tunnel. You don't need any DNS record or open ports.

Keep `MAX_UPLOAD_MB` under 100, because Cloudflare's free plan rejects larger uploads.

### Privacy

- Uploaded audio is deleted as soon as it has been analysed.
- Analysing needs no account. Without one, results (report, JSON, MIDI) are kept for `JOB_TTL_HOURS` under an
  unguessable link that anyone who has it can open.
- With a free account (username + password, no email), analyses are saved to a private **History** until
  deleted, and a **Profile** page sums up the keys, tempos, modes, chords and moods across them. A saved
  analysis opens only for its owner. Accounts live in `users.db` in the data volume; passwords are stored hashed.

## Updating the server and checking it works

After you push changes, run this on the server:

```bash
cd ~/VMP && bash deploy.sh
```

It pulls the latest code, rebuilds, waits until the app is healthy, then runs a **self-test inside the server**. The self-test analyzes music with known answers (a 4/4 pop song, a 3/4 waltz, power chords, a key change, a phone-style `.m4a` clip, and a corrupt file) and sends one real request to your AI model. It ends with `RESULT: PASS` or `RESULT: FAIL` plus the reasons, so you don't need to download results to find out whether an update works.

- At any time, `https://your-site/health` shows the running version, whether the analysis worker is up, and whether AI is configured. No password is needed, and it shows nothing private.
- Every page footer and every report shows the analyzer version.

### Before shipping changes (developers)

```bash
python selftest.py --full                      # adds logic checks and a complete web-app run with a simulated AI
python selftest.py --facts my_facts.json       # plus facts you've confirmed about your own recordings
```

`my_facts.json` holds facts a musician has confirmed, from a score or by ear (tempo, meter, key, chords in order, bass presence, claims that must not be made). Each one becomes a permanent check, so a later change can't silently undo it. The file is git-ignored because it points to private recordings.

## Command line

```bash
python musicanalyze.py song1.mp3 song2.mp3
python musicanalyze.py ./music_folder --out ./analysis --no-llm
python musicanalyze.py song.m4a --melody "A E F E A E D E"
python musicanalyze.py ./music_folder --melody "copy 4: A E F E A E D E"   # one track only
```

- Accepts `.mp3 .m4a .wav .ogg .flac .aac`, as single files or whole folders.
- Writes `music_report.md`, `music_results.json`, `midi/*.mid` and `sheets/*` (MusicXML + engraved HTML) to `--out`.
- The AI brainstorm runs when an API key is set. On Windows PowerShell, set it with `$env:NVIDIA_API_KEY="nvapi-..."`, not `export`. Use `--no-llm` to skip it.
- Speed: about 20 s per minute of audio on CPU. Re-running the same recording (to add a melody or retry the AI) takes about 1 s, because the analysis is cached by file content. Use `--no-cache` to force a fresh analysis.
- **Pick the fastest AI model** for your API key (it times each configured model on a real prompt):

  ```bash
  python musicanalyze.py --bench-llm
  # on the server:  docker compose exec music-analyzer python musicanalyze.py --bench-llm
  ```

## How it saves time

- **Results first, AI second.** The web page shows the full analysis as soon as it's ready. The AI section fills in by itself when it arrives.
- **The AI never holds you up for long.** It gets a compact summary instead of the full data (about 2.5k characters, down from about 12k). If the main model hasn't answered after 30 s, a backup model is asked *in parallel*, and the first good answer wins. After 150 s it gives up and your results stay as they are. The results are already on screen by then, so the AI never holds you up.
- **Overlap.** With several tracks, the AI works on one track while the next is being analyzed.
- **Cache.** Uploading the same recording again skips the analysis (seconds instead of a minute).
- **A warm worker.** One background worker process loads everything once at start-up, so uploads don't each pay that ~15–30 s start-up cost. If a job crashes it or runs out of memory, only that job fails, and a fresh worker takes over.

## Reading the report

The report explains itself. Underlined terms show a plain-English explanation on hover (tap on a phone), and the **📖 How to read this report** section at the top covers every column (Camelot, Mode, Form, Energy, the drum grid and so on).

## How accurate is it?

- **Key:** the tool compares the song's pitch-class energy against all 24 major/minor key profiles (Krumhansl-Schmuckler), using a 10-second window every 2 seconds. On real music, expect about 70–80% accuracy. The usual mistakes are relative keys (B minor ↔ D major) or the dominant key. Check the **runner-up** key when the result looks wrong.
- **Key-change times** are accurate to about ±4 s. Key sections shorter than 8 s are merged into the surrounding key, so a passing chord isn't counted as a modulation. To catch shorter changes, adjust `STEP`, `WINDOW`, and `MIN_SECTION` at the top of the script.
- **Tempo** can come out at half or double the felt tempo (for example 70 vs 140).
- **Chords** are detected once per beat, using major, minor, 7 and m7 shapes only. Detection works best on clear, sustained harmony. Dense mixes, distortion and extended jazz chords will be simplified or misread. Two related chords can be confused when they share notes (for example B and G#m).
- **Mode** is guessed from how strongly characteristic notes appear (for example the major 6th means Dorian). Treat it as a hint.
- **Drums** are not transcribed hit by hit. The tool averages where kick-heavy and snare-heavy hits land across all bars, which is robust on demos but assumes 4/4. Demucs (optional) separates the drums more cleanly than the built-in split.
- **Bass** is pitch-tracked from the lowest line once the drums are removed. If another low instrument (left-hand piano, low guitar strings) plays along, it's counted as bass.
- **Melody** can't be reliably extracted automatically from a full-band phone recording, because the chords and any held notes dominate. That's why you can type it. The check against the recording confirms which notes are present. It often can't confirm their order, and it says so.
- **Sections** come from automatic segmentation. The letters are a rough guide to which parts repeat.
- Treat everything as a starting point for ideas, not as a transcription.
