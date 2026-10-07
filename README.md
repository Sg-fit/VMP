# 🎵 Music Inspiration Analyzer

Analyzes instrumental tracks (no vocals needed) and turns the analysis into brainstorming material for musicians.

| Section | What you get |
|---|---|
| **Harmony** | Key and [Camelot](https://mixedinkey.com/camelot-wheel/) code, key timeline, time spent in each key, **mode** (Dorian, harmonic minor, Mixolydian…) with its signature "colour note", the notes used most, the chords used most, harmonic rhythm, **repeating chord loops** with Roman numerals (for example `Bm – Em – F#` = i – iv – V), and the full chord timeline |
| **Structure** | Sections (A, B, A…) with times, energy level, key and main chords |
| **Instruments** | Drums separated from the rest: the main drum groove as a 16-step grid (kick/snare/cymbal), how repetitive it is, likely fills, and which sections have drums. Bass line: main notes, range, style (pedal, roots, riff, walking), how often it plays the chord root, and slash chords (A/E) in the chord timeline |
| **Groove & sound** | Pulse steadiness, how busy the rhythm is, syncopation, percussive vs. tonal balance, tone and texture, loudness, energy arc, biggest build, quietest moment, frequency balance from sub to air |
| **🎶 Melody** | Type the melody or riff you hear (e.g. `A E F E A E D E`) to get its scale degrees, which modes it fits, its shape, melodic devices, chords that harmonize each note, and generated variations (inversion, retrograde, sequence, answer phrase) as MIDI files. Each note is also **checked against the recording**: it flags notes that barely sound, and says whether the audio supports the order. There's also an automatic "top line" that says honestly when it's only following the chords |
| **🎧 Shared DNA** | Building blocks the track shares with well-known recordings: chord loops matched against famous progressions (for example i–IV = Santana's “Oye Como Va”), the mode, the drum groove, and melodic devices. The AI adds more reference tracks, labeled as suggestions to double-check |
| **💡 Ideas to try** | Rule-based suggestions that need no AI: how to use the mode's colour note, chords to borrow from the parallel key, smooth and dramatic key-change targets, half-time or double-time variants, breakdowns and builds, groove contrasts, gaps in the mix |
| **🤖 AI brainstorm** (optional) | A description of the feel, 6–8 specific ideas tied to times and chords, 2–3 alternative chord progressions, and title ideas |
| **🎹 MIDI sketches** | `.mid` files of each detected chord loop, a variation of each loop, and the AI's progressions. Drag them into any DAW. |
| **🔗 Tracks that blend well** | With several files: which tracks are key-compatible (on the Camelot wheel) and close in tempo, for medleys, mashups or set lists |

It comes as a **web app** (upload tracks in the browser) and a **command-line tool**. Only the analysis numbers are sent to the AI, never the audio.

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
cp .env.example .env        # then edit .env: API key, APP_PASSWORD
docker compose up -d --build
```

The app listens on **127.0.0.1:8090** (only reachable from the server itself; change `HOST_PORT` if 8090 is taken). Put your reverse proxy or Cloudflare Tunnel in front of it, see below. Job data lives in the `music-data` volume and is deleted automatically after `JOB_TTL_HOURS`.

### Option B: plain Python (Linux)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY=nvapi-...   APP_PASSWORD=choose-one
gunicorn --workers 1 --threads 4 --timeout 120 --bind 127.0.0.1:8090 app:app
```

Use **one** worker. Analysis runs in a background thread inside that process, so extra workers wouldn't see each other's jobs. To keep it running, wrap the command in a systemd service.

### Settings (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `NVIDIA_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | — | Turns on the AI brainstorm (optional) |
| `MUSIC_MODEL` | provider default | Use a different model |
| `APP_PASSWORD` | — | If set, the site asks for this password (any username works). **Recommended on a public server**, so strangers can't use up your CPU and AI quota. |
| `MAX_UPLOAD_MB` | 95 | Max upload size per request (Cloudflare's free plan caps uploads at 100 MB) |
| `MAX_FILES` | 10 | Max tracks per upload |
| `JOB_TTL_HOURS` | 24 | How long results are kept |
| `DATA_DIR` | `./jobs` | Where results are stored |
| `HOST_PORT` | 8090 | Docker only: the localhost port your proxy forwards to |
| `MUSIC_SEPARATION` | `auto` | `auto` uses Demucs if installed, `demucs` requires it, `simple` always uses the fast built-in split |
| `MUSIC_MODEL` / `MUSIC_FALLBACK_MODELS` | provider default | Main AI model, and backups that are raced in if it's slow or fails. Run `--bench-llm` (below) to pick the fastest |
| `MUSIC_AI_HEDGE_SECONDS` / `MUSIC_AI_DEADLINE_SECONDS` | 15 / 60 | Ask a backup model in parallel after this many seconds; give up on AI ideas after this many |
| `MUSIC_LLM_BASE_URL` | NVIDIA's API | Any OpenAI-compatible endpoint, for example a self-hosted model |
| `MUSIC_CACHE_DIR` | `DATA_DIR/cache` (web), `~/.cache/music-analyzer` (CLI) | Cached analyses. A re-upload of the same recording skips the slow analysis. Entries unused for 30 days are deleted |
| `CPU_LIMIT` / `MEM_LIMIT` | 1.0 / 1500m | Docker only: caps on CPU cores and memory for the analyzer, so jobs can't slow down other sites on the server. A job that needs more memory fails with a message instead of freezing the machine |
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
- Results (report, JSON, MIDI) are kept for `JOB_TTL_HOURS`, under an unguessable link.
- Anyone who has a results link can open it, so set `APP_PASSWORD` if that matters.

## Command line

```bash
python musicanalyze.py song1.mp3 song2.mp3
python musicanalyze.py ./music_folder --out ./analysis --no-llm
python musicanalyze.py song.m4a --melody "A E F E A E D E"
python musicanalyze.py ./music_folder --melody "copy 4: A E F E A E D E"   # one track only
```

- Accepts `.mp3 .m4a .wav .ogg .flac .aac`, as single files or whole folders.
- Writes `music_report.md`, `music_results.json` and `midi/*.mid` to `--out`.
- The AI brainstorm runs when an API key is set. On Windows PowerShell, set it with `$env:NVIDIA_API_KEY="nvapi-..."`, not `export`. Use `--no-llm` to skip it.
- Speed: about 20 s per minute of audio on CPU. Re-running the same recording (to add a melody or retry the AI) takes about 1 s, because the analysis is cached by file content. Use `--no-cache` to force a fresh analysis.
- **Pick the fastest AI model** for your API key (it times each configured model on a real prompt):

  ```bash
  python musicanalyze.py --bench-llm
  # on the server:  docker compose exec music-analyzer python musicanalyze.py --bench-llm
  ```

## How it saves time

- **Results first, AI second.** The web page shows the full analysis as soon as it's ready. The AI section fills in by itself when it arrives.
- **The AI never holds you up for long.** It gets a compact summary instead of the full data (about 2.5k characters, down from about 12k). If the main model hasn't answered after 15 s, a backup model is asked *in parallel*, and the first good answer wins. After 60 s it gives up and your results stay as they are.
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
