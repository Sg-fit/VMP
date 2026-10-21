#!/usr/bin/env python3
"""Tags for finding new music: describe a set of songs (uploads, a playlist link, or a typed list) in the words
music sites search by, then turn those words into searches that reach past the most popular results.

Where the tags come from, and how much each counts
- Songs we have the audio for (weight 1.0): our own light analysis (tempo, metre, key/mode, energy, mood,
  drums or not, blues form) plus, when it's set up, a trained style model (400 Discogs styles, moods, instruments,
  vocals or not; see tagger.py).
- Songs we only know by name (weight 0.4): looked up on MusicBrainz (and Last.fm when LASTFM_API_KEY is set).
  Those are other people's tags for the artist or the song, not a measurement of this recording, so they
  count for less.
- A tag's score is the weighted share of songs it describes. Broad tags ("rock", "electronic") are listed but
  left out of the searches: they lead straight back to the biggest names.

Network use: Spotify's public embed page (playlist names and artists only), YouTube via yt-dlp (titles only),
MusicBrainz (1 request a second, as they ask), Last.fm (optional). MUSIC_OFFLINE=1 turns all of it off.
"""

import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

AUDIO_WEIGHT, NAME_WEIGHT = 1.0, 0.4
MAX_SONGS = 100
USER_AGENT = "MusicInspirationAnalyzer/1.0 (https://github.com/Sg-fit/VMP)"
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

# Top-level genres (Discogs' parents and the common crowd-tag equivalents): useful as a summary, useless for
# finding something new, so they're never used as searches on their own.
BROAD = {"blues", "brass & military", "children's", "classical", "electronic", "folk, world, & country",
         "folk world & country", "funk / soul", "funk", "soul", "hip hop", "hip-hop", "rap", "jazz", "latin",
         "non-music", "pop", "reggae", "rock", "stage & screen", "metal", "country", "folk", "indie", "alternative",
         "dance", "rnb", "r&b", "electronica", "world", "instrumental", "singer-songwriter", "soundtrack",
         "alternative rock", "indie rock", "pop rock", "classic rock", "hard rock", "experimental"}
# Crowd tags that say nothing about the sound.
JUNK = re.compile(r"^(seen live|favou?rites?|my |love|loved|awesome|beautiful|best|good|great|cool|amazing|"
                  r"spotify|youtube|albums? i own|under \d+|\d{4}$|check out|to listen|wishlist|playlist)", re.I)
PLACES = {"british", "american", "uk", "usa", "english", "german", "french", "japanese", "canadian", "swedish",
          "australian", "irish", "scottish", "italian", "spanish", "korean", "brazilian", "norwegian", "dutch",
          "finnish", "danish", "icelandic", "belgian", "polish", "russian", "mexican", "chinese", "new zealand",
          "england", "scotland", "wales", "london", "usa", "united states", "united kingdom", "lgbtq+", "lgbt", "queer"}


def key(tag):
    """One spelling per tag: 'Trip-Hop', 'trip hop' and 'triphop' are the same tag."""
    return re.sub(r"[^a-z0-9&]+", "", tag.lower().replace("and", "&"))


# ---------------------------------------------------------------- network helpers
_last_call = defaultdict(float)


def offline():
    return os.environ.get("MUSIC_OFFLINE") == "1"


def _get(url, host_gap=0.0, ua=USER_AGENT, timeout=20, retries=2, data=None):
    """GET url -> text. host_gap: minimum seconds between calls to the same host (MusicBrainz asks for 1).
    A busy server (503/429) is retried after a pause."""
    import urllib.error
    if offline():
        raise OSError("offline")
    host = urllib.parse.urlparse(url).netloc
    for attempt in range(retries + 1):
        wait = _last_call[host] + host_gap - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            headers = {"User-Agent": ua, "Accept-Language": "en"}
            if data is not None:  # a JSON query sent as POST (ListenBrainz popularity)
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == retries:
                raise
            time.sleep(2 * (attempt + 1))
        finally:
            _last_call[host] = time.time()


class Cache:
    """Lookups kept on disk for 30 days, so the same artist or playlist isn't fetched twice."""
    def __init__(self, folder):
        self.dir = Path(folder) / "lookups" if folder else None

    def get(self, name, fetch, days=30):
        f = self.dir / (hashlib.sha1(name.encode()).hexdigest()[:24] + ".json") if self.dir else None
        if f and f.exists() and time.time() - f.stat().st_mtime < days * 86400:
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        value = fetch()
        if f and value is not None:
            try:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(value), encoding="utf-8")
            except OSError:
                pass
        return value


# ---------------------------------------------------------------- reading the song names
NOISE = re.compile(r"\s*[\(\[][^)\]]*(official|video|audio|lyrics?|visuali[sz]er|hd|hq|4k|remaster|live at|"
                   r"full album|music video|mv)[^)\]]*[\)\]]", re.I)


FEAT = re.compile(r"\s*[\(\[]?\b(ft|feat|featuring)\b\.?\s.*$", re.I)


def main_artist(a):
    """'Peso Pluma, Anitta' / 'A x B' / 'A feat. B' -> the first-named artist (what lookups match best)."""
    a = FEAT.sub("", a or "")
    return re.split(r"\s*(?:,|;| x |/)\s*", a)[0].strip() if a else a


def clean_title(t):
    t = FEAT.sub("", NOISE.sub("", t or ""))
    t = re.sub(r"\s*\|.*$", "", t)  # "Song | Channel name"
    return re.sub(r"\s+", " ", t).strip(" -–—")


def split_name(line, default_artist=None):
    """'Artist - Title', 'Title by Artist', '1. Artist – Title' -> (artist, title)."""
    line = re.sub(r"^\s*(\d+[\.\)]|[-*•])\s*", "", line.strip())
    line = clean_title(line)
    if not line:
        return None
    for sep in (" - ", " – ", " — ", " -- ", " ~ "):
        if sep in line:
            a, t = line.split(sep, 1)
            a, t = a.strip(), clean_title(t)
            # YouTube titles are sometimes "Title - Artist": the channel name tells which side is the artist
            if default_artist and key(default_artist) in key(t) and key(default_artist) not in key(a):
                a, t = t, a
            return {"artist": main_artist(a), "title": t}
    m = re.match(r"(.+?)\s+by\s+(.+)$", line, re.I)
    if m:
        return {"artist": main_artist(m.group(2).strip()), "title": m.group(1).strip()}
    return {"artist": default_artist, "title": line}


def parse_song_list(text):
    songs = [s for s in (split_name(line) for line in (text or "").splitlines()) if s]
    return songs[:MAX_SONGS]


def spotify_id(url):
    m = re.search(r"(?:open\.spotify\.com/(?:intl-[a-z]+/)?|spotify:)(playlist|album|track)[/:]([A-Za-z0-9]+)", url)
    return (m.group(1), m.group(2)) if m else None


def parse_spotify_embed(html):
    """The public embed page carries the list as JSON in <script id="__NEXT_DATA__">."""
    m = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise ValueError("Spotify didn't show us the song list. The playlist may be private")
    ent = json.loads(m.group(1))["props"]["pageProps"]["state"]["data"]["entity"]
    name = ent.get("name") or ent.get("title") or "Spotify list"
    tracks = ent.get("trackList") or []
    if not tracks and ent.get("title"):  # a single track
        tracks = [{"title": ent["title"], "subtitle": ", ".join(a.get("name", "") for a in ent.get("artists", []))}]
    songs = [{"artist": main_artist((t.get("subtitle") or "").replace(" ", " ")) or None,
              "title": clean_title(t.get("title", ""))} for t in tracks if t.get("title")]
    return name, songs[:MAX_SONGS]


def read_playlist(url, cache):
    """Spotify playlist/album/track or YouTube playlist link -> (list name, [{artist, title}])."""
    url = (url or "").strip()
    sp = spotify_id(url)
    if sp:
        kind, sid = sp
        html = cache.get(f"spotify:{kind}:{sid}", lambda: _get(f"https://open.spotify.com/embed/{kind}/{sid}",
                                                              ua=BROWSER_UA), days=1)
        return parse_spotify_embed(html)
    if re.search(r"(youtube\.com|youtu\.be)/", url):
        return cache.get(f"youtube:{url}", lambda: _youtube_list(url), days=1)
    raise ValueError("that doesn't look like a Spotify or YouTube playlist link")


def _youtube_list(url):
    if offline():
        raise OSError("offline")
    import yt_dlp
    opts = {"extract_flat": "in_playlist", "quiet": True, "no_warnings": True, "skip_download": True,
            "playlistend": MAX_SONGS}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    songs = []
    for e in (info.get("entries") or [info])[:MAX_SONGS]:
        channel = re.sub(r"\s*(- Topic|VEVO|Official)$", "", e.get("channel") or e.get("uploader") or "", flags=re.I)
        s = split_name(e.get("title") or "", default_artist=channel or None)
        if s:
            songs.append(s)
    return [info.get("title") or "YouTube playlist", songs]


def names_from_file(path, load_meta=None):
    """Artist and title for an uploaded file: its embedded tags, else 'Artist - Title.mp3'."""
    meta = {}
    try:
        import av
        with av.open(str(path)) as c:
            meta = {k.lower(): v for k, v in (c.metadata or {}).items()}
    except Exception:
        pass
    if meta.get("title") and meta.get("artist"):
        return {"artist": main_artist(meta["artist"]), "title": clean_title(meta["title"])}
    s = split_name(Path(path).stem)
    return s if s and s.get("artist") else None


# ---------------------------------------------------------------- tags by name (what others call the song)
def _good(tag, artist=None):
    t = tag.strip().lower()
    return (2 <= len(t) <= 40 and not JUNK.match(t) and t not in PLACES and
            not (artist and key(t) == key(artist)))


def _ranked(items, artist=None, n=8):
    merged = {}  # 'trip hop' and 'trip-hop' are one tag: keep the commoner spelling, add the counts
    for i in items:
        if _good(i["name"], artist):
            k = key(i["name"])
            name, c = merged.get(k, (i["name"].lower(), 0))
            merged[k] = (name, c + i.get("count", 1))
    items = list(merged.values())
    if not items:
        return {}
    top = max(c for _, c in items) or 1
    return {name: round(min(1.0, c / top), 2) for name, c in sorted(items, key=lambda x: -x[1])[:n]
            if c >= max(1, 0.15 * top)}


def musicbrainz(song, cache):
    """{tag: 0-1} from MusicBrainz: the recording's own genres/tags if it has any, else the artist's."""
    artist, title = song.get("artist"), song.get("title")
    if not title:
        return {}
    q = lambda s: '"' + s.replace('"', "") + '"'

    def find():
        query = f"recording:{q(title)}" + (f" AND artist:{q(artist)}" if artist else "")
        data = json.loads(_get("https://musicbrainz.org/ws/2/recording?fmt=json&limit=3&query=" +
                               urllib.parse.quote(query), host_gap=1.1))
        recs = [r for r in data.get("recordings", []) if int(r.get("score", 0)) >= 85]
        if not recs:
            return {}
        r = recs[0]
        credit = (r.get("artist-credit") or [{}])[0].get("artist", {})
        return {"recording_tags": r.get("tags", []), "artist_id": credit.get("id"), "artist": credit.get("name")}
    hit = cache.get(f"mb:rec:{artist}|{title}", find)
    if not hit:
        return {}
    tags = _ranked(hit.get("recording_tags", []), hit.get("artist"))
    if len(tags) < 2 and hit.get("artist_id"):
        aid = hit["artist_id"]
        a = cache.get(f"mb:artist:{aid}", lambda: json.loads(_get(
            f"https://musicbrainz.org/ws/2/artist/{aid}?inc=genres+tags&fmt=json", host_gap=1.1)))
        # official genres first; free-form tags fill in descriptors ("melancholic", "female vocalists")
        tags = {**_ranked(a.get("tags", []), hit.get("artist")), **_ranked(a.get("genres", []), hit.get("artist"))}
    song["found_artist"] = hit.get("artist")
    song["artist_id"] = hit.get("artist_id")
    return tags


def lastfm(song, cache):
    k = os.environ.get("LASTFM_API_KEY")
    if not k or not song.get("artist") or not song.get("title"):
        return {}
    base = "https://ws.audioscrobbler.com/2.0/?format=json&autocorrect=1&api_key=" + urllib.parse.quote(k)

    def track():
        return json.loads(_get(base + "&method=track.gettoptags&artist=" + urllib.parse.quote(song["artist"]) +
                               "&track=" + urllib.parse.quote(song["title"]), host_gap=0.25))
    data = cache.get(f"lfm:track:{song['artist']}|{song['title']}", track)
    items = [{"name": t["name"], "count": int(t.get("count", 0))} for t in (data.get("toptags") or {}).get("tag", [])]
    if len(items) < 3:
        def artist():
            return json.loads(_get(base + "&method=artist.gettoptags&artist=" + urllib.parse.quote(song["artist"]),
                                   host_gap=0.25))
        data = cache.get(f"lfm:artist:{song['artist']}", artist)
        items = [{"name": t["name"], "count": int(t.get("count", 0))} for t in (data.get("toptags") or {}).get("tag", [])]
    return _ranked(items, song["artist"])


def lookup(song, cache):
    """Tags others give this song or its artist: {tag: 0-1}, plus where they came from."""
    tags, sources = {}, []
    for name, fn in (("Last.fm", lastfm), ("MusicBrainz", musicbrainz)):
        try:
            got = fn(song, cache)
        except (OSError, ValueError, KeyError) as e:
            song.setdefault("lookup_errors", []).append(f"{name}: {str(e)[:80]}")
            continue
        if got:
            sources.append(name)
            for t, v in got.items():
                tags[t] = max(tags.get(t, 0), v)
    song["sources"] = sources
    return tags


# ---------------------------------------------------------------- tags from the audio
TEMPO_BANDS = [(70, "very slow"), (90, "slow"), (110, "mid-tempo"), (130, "upbeat"), (155, "fast"), (999, "very fast")]


def measured_tags(r):
    """Describe one analysed song in searchable words: {category: {tag: 0-1}}."""
    out = {"mood": {}, "sound": {}, "rhythm": {}, "style": {}}
    bpm = r.get("tempo_bpm")
    if bpm:
        band = next(name for top, name in TEMPO_BANDS if bpm < top)
        out["rhythm"][band] = 0.6 if r.get("tempo_approximate") else 1.0
    else:
        out["rhythm"]["beatless"] = 1.0
    m = r.get("meter") or {}
    if m.get("beats_per_bar") == 3 and m.get("confidence") == "detected":
        out["rhythm"]["3/4 time"] = 1.0
    elif m.get("beats_per_bar") in (5, 7):
        out["rhythm"]["odd time signature"] = 1.0
    rh = r.get("rhythm") or {}
    if rh.get("groove") == "heavily syncopated":
        out["rhythm"]["syncopated"] = 0.8
    if (rh.get("tempo_feel") or "").startswith("free"):
        out["rhythm"]["rubato"] = 0.8
    mode, conf = r.get("mode"), r.get("mode_confidence")
    if mode in ("Dorian", "Phrygian", "Lydian", "Mixolydian") and conf != "low":
        out["sound"][mode.lower() + " mode"] = 0.7
    if "minor" in (r.get("key") or ""):
        out["mood"]["minor key"] = 0.5  # (tag mode reads keys from a quicker, rougher split)
    f = r.get("feel") or {}
    for w in (f.get("mood") or "").split(","):
        if w.strip():
            out["mood"][w.strip()] = 0.8
    if f.get("energy"):
        if f["energy"] <= 3:
            out["mood"]["mellow"] = 0.8
        elif f["energy"] >= 7:
            out["mood"]["high energy"] = 0.8
    drums, bass = (r.get("drums") or {}).get("present"), (r.get("bass") or {}).get("present")
    if drums is False and bass is False:
        out["sound"]["sparse"] = 0.7
    s = r.get("sound") or {}
    if s.get("tone") == "dark/warm":
        out["sound"]["warm"] = 0.6
    elif s.get("tone") == "bright":
        out["sound"]["bright"] = 0.5
    if s.get("texture") == "noisy / textural":
        out["sound"]["textural"] = 0.6
    if r.get("blues_form"):
        out["style"][r["blues_form"]["name"].replace(" minor", "")] = 1.0
    return out


def model_tags(m):
    """tagger.summarise() output -> {category: {tag: 0-1}} and a broad genre for each style."""
    out = {"style": {}, "genre": {}, "mood": {}, "sound": {}}
    for label, p in m.get("styles", []):
        parent, _, style = label.partition("---")
        out["style"][style.lower()] = max(out["style"].get(style.lower(), 0), p)
        out["genre"][parent.lower()] = max(out["genre"].get(parent.lower(), 0), p)
    for mood, p in m.get("moods", []):
        out["mood"][mood.lower()] = p
    for inst, p in m.get("instruments", []):  # shown, not searched: "guitar" alone finds everything
        out.setdefault("instruments", {})[inst.lower()] = p
    v = m.get("voice")
    if v is not None:
        if v < 0.25:
            out["sound"]["instrumental"] = round(1 - v, 2)
        elif v > 0.6 and m.get("gender") is not None:
            g = m["gender"]
            if g > 0.75:
                out["sound"]["female vocals"] = round(g, 2)
            elif g < 0.25:
                out["sound"]["male vocals"] = round(1 - g, 2)
    for k, word in (("acoustic", "acoustic"), ("electronic", "electronic"), ("danceable", "danceable"),
                    ("aggressive", "aggressive"), ("relaxed", "relaxed"), ("sad", "sad"), ("happy", "happy")):
        p = m.get(k)
        if p is not None and p >= 0.7:
            (out["sound"] if k in ("acoustic", "electronic", "danceable") else out["mood"])[word] = round(p, 2)
    return out


def name_tags(tags):
    """Crowd tags -> categories. Genre words go to style; everything else is a mood/sound descriptor."""
    out = {"style": {}, "genre": {}, "mood": {}, "sound": {}}
    moodish = re.compile(r"(chill|mellow|sad|melanchol|dark|happy|calm|relax|dream|atmospher|moody|energetic|"
                         r"romantic|nostalg|haunting|uplifting|aggressive|emotional|beautiful|sexy|epic)", re.I)
    soundish = re.compile(r"(vocal|instrumental|acoustic|piano|guitar|saxophone|strings|synth|lo-?fi|bass|"
                          r"\d0s$|female|male)", re.I)
    for t, v in tags.items():
        if t in BROAD:
            out["genre"][t] = v
        elif moodish.search(t):
            out["mood"][t] = v
        elif soundish.search(t):
            out["sound"][t] = v
        else:
            out["style"][t] = v
    return out


# ---------------------------------------------------------------- putting it together
def aggregate(songs):
    """songs: [{weight, tags: {category: {tag: score}}}] -> {category: [{tag, score, songs, from}]} sorted.
    score = weighted share of the list the tag describes (0-1)."""
    total = sum(s["weight"] for s in songs) or 1
    acc = defaultdict(lambda: {"score": 0.0, "songs": 0, "names": defaultdict(float), "from": set()})
    for s in songs:
        for cat, tags in s["tags"].items():
            for t, v in tags.items():
                a = acc[(cat if t.lower() not in BROAD else "genre", key(t))]
                a["score"] += s["weight"] * v
                a["songs"] += 1
                a["names"][t] += v
                a["from"].update(s.get("from", []))
    out = defaultdict(list)
    for (cat, _), a in acc.items():
        out[cat].append({"tag": max(a["names"], key=a["names"].get), "score": round(a["score"] / total, 3),
                         "songs": a["songs"], "from": sorted(a["from"])})
    return {cat: sorted(v, key=lambda x: (-x["score"], x["tag"])) for cat, v in out.items()}


def slug(tag):
    return re.sub(r"[^a-z0-9]+", "-", tag.lower().replace("&", "and")).strip("-")


def links(tag, genre=True):
    q = urllib.parse.quote
    out = {"Bandcamp": f"https://bandcamp.com/tag/{slug(tag)}",
           "Last.fm": f"https://www.last.fm/tag/{q(tag.lower())}/artists",
           "MusicBrainz": f"https://musicbrainz.org/tag/{q(tag.lower())}"}
    if genre:
        out["RYM"] = f"https://rateyourmusic.com/genre/{slug(tag)}/"
    return out


def search_plan(agg, known_artists, n=8):
    """The tags worth searching by (specific, shared by several songs) and two-word combinations of them."""
    styles = [t for t in agg.get("style", []) if t["score"] >= 0.08][:n]
    moods = [t for t in agg.get("mood", []) + agg.get("sound", []) if t["score"] >= 0.15
             and t["tag"] not in ("minor key", "instrumental")]
    moods.sort(key=lambda t: -t["score"])
    q = urllib.parse.quote
    combos = []
    for s in styles[:3]:
        for m in moods[:2]:
            phrase = f"{s['tag']} {m['tag']}"
            combos.append({"query": phrase, "youtube": f"https://www.youtube.com/results?search_query={q(phrase)}",
                           "spotify": f"https://open.spotify.com/search/{q(phrase)}"})
    return {"styles": [{**t, "links": links(t["tag"])} for t in styles],
            "moods": [{**t, "links": links(t["tag"], genre=False)} for t in moods[:6]],
            "combos": combos[:6], "skip_artists": sorted(known_artists)}


# Lesser-known artists: real acts (some listeners) that are clearly less played than what the list already holds.
MIN_LISTENERS = 20                # fewer than this on ListenBrainz and there's little sign anyone plays them yet
MAX_LISTENERS = 10000             # the ceiling when we don't know how popular the list's own artists are
OWN_SHARE = 0.25                  # otherwise: under a quarter of the list's typical (median) artist


def artists_with_tags(tags, cache):
    """MusicBrainz artists carrying all of `tags` (one search; 1 request a second, as MusicBrainz asks)."""
    query = " AND ".join(f'tag:"{t.replace(chr(34), "")}"' for t in tags)

    def fetch():
        data = json.loads(_get("https://musicbrainz.org/ws/2/artist?fmt=json&limit=100&query=" + urllib.parse.quote(query),
                               host_gap=1.1))
        clean = lambda s: s.strip().strip("\"'").strip().lower()  # some listeners type tags in quotes
        return [{"id": a["id"], "name": a["name"], "tags": {clean(t["name"]): t.get("count", 1) for t in a.get("tags", [])}}
                for a in data.get("artists", [])]
    return cache.get(f"mb:artists:{query}", fetch, days=7) or []


def listener_counts(mbids, cache):
    """Distinct listeners per artist from ListenBrainz (public, no key). {mbid: count or None}."""
    out = {}
    todo = [m for m in dict.fromkeys(mbids) if m]
    for i in range(0, len(todo), 100):
        part = todo[i:i + 100]

        def fetch(part=part):
            rows = json.loads(_get("https://api.listenbrainz.org/1/popularity/artist", host_gap=0.5,
                                   data=json.dumps({"artist_mbids": part}).encode()))
            return {r["artist_mbid"]: r.get("total_user_count") for r in rows}
        out.update(cache.get("lb:pop:" + hashlib.sha1("|".join(part).encode()).hexdigest(), fetch, days=7) or {})
    return out


def lesser_known(agg, known, cache, extra_tags=(), n=12):
    """Artists with the list's sound that fewer people listen to. Searched by the list's most specific tags (pairs
    first: two tags together narrow the field), minus the artists already in the list, kept only between
    MIN_LISTENERS and a ceiling below the list's own artists, and ranked by how many of the list's tags they carry.
    known: {name: mbid or None}. Returns (artists, the tags searched)."""
    weight = {key(t["tag"]): t["score"] for cat in ("style", "mood", "sound") for t in agg.get(cat, [])}
    for t in extra_tags:
        weight[key(t)] = max(weight.get(key(t), 0), 1.0)
    seeds = list(dict.fromkeys(list(extra_tags) + [t["tag"] for t in agg.get("style", []) if t["score"] >= 0.08]))[:3]
    if not seeds:
        return [], []
    queries = [seeds[i:j] for i in range(len(seeds)) for j in range(i + 2, len(seeds) + 1) if j - i == 2] or [seeds[:1]]
    queries += [[seeds[0]]]
    found = {}
    for q in queries:
        for a in artists_with_tags(q, cache):
            found.setdefault(a["id"], a)
    known_keys = {key(name) for name in known}
    cands = [a for a in found.values() if key(a["name"]) not in known_keys and a["id"] not in set(known.values())]
    counts = listener_counts([a["id"] for a in cands] + [m for m in known.values() if m], cache)
    own = sorted(c for m in known.values() if m and (c := counts.get(m)))
    ceiling = min(MAX_LISTENERS, OWN_SHARE * own[len(own) // 2]) if own else MAX_LISTENERS
    out = []
    for a in cands:
        c = counts.get(a["id"])
        if c is None or not MIN_LISTENERS <= c <= ceiling:
            continue
        shared = list({key(t): t for t in a["tags"] if key(t) in weight}.values())  # 'trip-hop' = 'trip hop'
        if not shared:
            continue
        out.append({"name": a["name"], "mbid": a["id"], "listeners": c, "shared_tags": shared[:5],
                    "match": round(sum(weight[key(t)] for t in shared), 3)})
    # Among equally good matches, the ones more people already enjoy first: all of them are well below the
    # list's own artists anyway, and a few hundred listeners says more about a record than a handful.
    out.sort(key=lambda x: (-x["match"], -x["listeners"]))
    q = urllib.parse.quote
    for x in out[:n]:
        x["links"] = {"MusicBrainz": f"https://musicbrainz.org/artist/{x['mbid']}",
                      "Bandcamp": f"https://bandcamp.com/search?q={q(x['name'])}",
                      "YouTube": f"https://www.youtube.com/results?search_query={q(x['name'])}"}
    return out[:n], seeds


def parse_description(text):
    """'melancholic, fingerpicked blues; lo-fi' -> ['melancholic', 'fingerpicked blues', 'lo-fi']: what the
    listener asks for, used as tags of their own."""
    parts = re.split(r"[,;\n]+|\band\b", (text or "").lower())
    return list(dict.fromkeys(p.strip(" .") for p in parts if 2 <= len(p.strip(" .")) <= 40))[:6]


def run_tags(files, out_dir, song_text=None, playlist_url=None, progress=None, cache_dir=None, context=None,
             description=None):
    """The tag job. Writes tags_report.md and tags.json into out_dir; returns the report text."""
    import musicanalyze as ma
    import tagger
    progress = progress or (lambda msg: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Cache(cache_dir)
    notes, songs = [], []
    list_name = None

    # 1. the names: typed list and/or playlist link
    named = parse_song_list(song_text)
    if playlist_url:
        progress("Reading your playlist…")
        try:
            list_name, from_link = read_playlist(playlist_url, cache)
            named += from_link
        except Exception as e:
            notes.append(f"We couldn't read that playlist link ({str(e)[:160]}). If you paste the song names "
                         "instead, we'll look them up one by one.")
    named = named[:MAX_SONGS]

    # 2. the audio: light analysis (+ the style model), matched to a listed song when the names agree
    analysed = []
    for i, f in enumerate(files, 1):
        f = Path(f)
        progress(f"[{i}/{len(files)}] Listening to {f.name}…")
        try:
            r = ma.analyze(f, cache_dir=cache_dir, context=context, light=True)
        except Exception as e:
            notes.append(f"We couldn't open {f.name} ({str(e)[:120]}). Try exporting it again as mp3 or wav.")
            continue
        analysed.append((f, r))
    model = {}
    if analysed and tagger.available():
        progress("Working out the styles…")
        try:
            model = tagger.predict([str(f) for f, _ in analysed], ma.load_audio)
        except Exception as e:
            notes.append(f"The style model didn't run this time ({str(e)[:160]}), so these tags come from our "
                         "own measurements and the song names.")
    elif analysed:
        notes.append("The style model isn't installed on this server, so for the songs you uploaded we've "
                     "worked from our own measurements and, where the file has them, the song names.")

    for f, r in analysed:
        song = names_from_file(f) or {"artist": None, "title": None}
        match = next((n for n in named if song.get("title") and key(n["title"]) == key(song["title"])), None)
        if match:
            named.remove(match)
            song = {**match, **{k: v for k, v in song.items() if v}}
        tags = measured_tags(r)
        m = model.get(str(f))
        if m:
            for cat, d in model_tags(m).items():
                tags.setdefault(cat, {}).update(d)
        froms = ["our analysis"] + (["style model"] if m else [])
        if song.get("title") and song.get("artist"):
            progress(f"Looking up {song['artist']} - {song['title']}...")
            crowd = lookup(song, cache)
            if crowd:  # others' tags count for less than what the audio itself showed
                for cat, d in name_tags(crowd).items():
                    for t, v in d.items():
                        tags.setdefault(cat, {})[t] = max(tags.get(cat, {}).get(t, 0), NAME_WEIGHT * v)
                froms += song.get("sources", [])
        songs.append({"name": _display(song, f.name), "weight": AUDIO_WEIGHT, "tags": tags, "from": froms,
                      "artist": song.get("found_artist") or song.get("artist"), "audio": True,
                      "artist_id": song.get("artist_id"),
                      "measured": {"tempo_bpm": r.get("tempo_bpm"),
                                   "mood": (r.get("feel") or {}).get("mood"),
                                   "styles": [s for s, _ in (m or {}).get("styles", [])[:3]]}})

    # 3. songs known only by name
    for i, s in enumerate(named, 1):
        progress(f"Looking up song {i} of {len(named)}: {_display(s)}…")
        crowd = lookup(s, cache)
        songs.append({"name": _display(s), "weight": NAME_WEIGHT, "tags": name_tags(crowd),
                      "from": s.get("sources", []), "artist": s.get("found_artist") or s.get("artist"),
                      "artist_id": s.get("artist_id"), "audio": False, "found": bool(crowd)})
    if named and offline():
        notes.append("Looking songs up by name is switched off on this server, so we could only use the songs "
                     "you uploaded.")
    missing = [s["name"] for s in songs if not s["audio"] and not s.get("found")]
    if missing:
        notes.append(f"We couldn't find tags for {len(missing)} of the songs you named: " + ", ".join(missing[:8]) +
                     (" ..." if len(missing) > 8 else "") + ". Very new or little-known songs often aren't listed "
                     "yet, and a typo in the name can hide one.")

    usable = [s for s in songs if s["audio"] or s.get("found")]
    agg = aggregate(usable)
    asked = parse_description(description)
    if asked:  # the listener's own words: shown as theirs, searched first
        agg["asked"] = [{"tag": t, "score": 1.0, "songs": 0, "from": ["you"]} for t in asked]
    known = {s["artist"]: s.get("artist_id") for s in songs if s.get("artist") and (s["audio"] or s.get("found"))}
    plan = search_plan(agg, set(known))
    for t in reversed(asked):  # your words lead the searches too
        plan["styles"].insert(0, {"tag": t, "score": 1.0, "songs": 0, "from": ["you"], "links": links(t)})
    artists, searched = [], []
    if not offline() and (agg.get("style") or asked):
        progress("Looking for lesser-known artists with this sound…")
        try:
            artists, searched = lesser_known(agg, known, cache, extra_tags=asked)
        except (OSError, ValueError, KeyError) as e:
            notes.append(f"We couldn't look for lesser-known artists this time ({str(e)[:120]}).")
    data = {"list_name": list_name, "songs": songs, "tags": agg, "search": plan, "notes": notes,
            "lesser_known": artists, "searched_tags": searched,
            "weights": {"audio": AUDIO_WEIGHT, "name_only": NAME_WEIGHT},
            "model": tagger.backend(), "credit": tagger.CREDIT if model else None}
    (out_dir / "tags.json").write_text(json.dumps(data, indent=2, default=list), encoding="utf-8")
    report = tag_report(data)
    (out_dir / "tags_report.md").write_text(report, encoding="utf-8")
    if cache_dir:  # old analyses and lookups go after a while
        ma.prune_cache(cache_dir)
        for f in (Path(cache_dir) / "lookups").glob("*.json"):
            try:
                if f.stat().st_mtime < time.time() - 30 * 86400:
                    f.unlink()
            except OSError:
                pass
    return data, report


def _display(s, fallback=None):
    if s.get("artist") and s.get("title"):
        return f"{s['artist']} - {s['title']}"
    return s.get("title") or fallback or "?"


def _fmt(tags, n=8):
    return ", ".join(f"{t['tag']} ({round(100 * t['score'])}%)" for t in tags[:n]) or "—"


def tag_report(d):
    L = ["# Tags for finding new music", ""]
    n_audio = sum(s["audio"] for s in d["songs"])
    n_name = len(d["songs"]) - n_audio
    head = f"**{d['list_name']}** — " if d.get("list_name") else ""
    plural = lambda n, w: f"{n} {w}{'' if n == 1 else 's'}"
    counts = " and ".join(x for x in (n_audio and plural(n_audio, "song") + " listened to",
                                      n_name and plural(n_name, "song") + " looked up by name") if x)
    L.append(f"{head}{counts}. Each percentage shows how much of your list a tag describes" +
             (f". Songs we listened to count {AUDIO_WEIGHT:g} and songs we only know by name count {NAME_WEIGHT:g}, "
              "because their tags are other listeners' words for the artist rather than something we heard."
              if n_audio and n_name else
              ". Tags for songs we only know by name come from other listeners, mostly describing the artist."
              if n_name else "."))
    L.append("")
    for note in d["notes"]:
        L.append(f"> {note}")
    if d["notes"]:
        L.append("")
    t = d["tags"]
    if not any(t.values()):
        L.append("We couldn't come up with any tags this time. Try uploading the songs, or check the names are "
                 "spelled the way Spotify spells them.")
        return "\n".join(L)
    L += ["## Your tags", ""]
    if t.get("asked"):
        L.append("- **You asked for:** " + ", ".join(x["tag"] for x in t["asked"]))
    for cat, label, n in (("style", "Styles", 8), ("mood", "Mood", 8), ("sound", "Sound", 8), ("rhythm", "Rhythm", 8),
                          ("instruments", "Instruments heard", 6),
                          ("genre", "Broad genres", 5)):
        if t.get(cat) or cat == "style":
            note = " (we leave these out of the searches, because they lead straight back to the biggest names)" \
                if cat == "genre" else ""
            L.append(f"- **{label}**{note}: {_fmt(t.get(cat, []), n)}")
    L.append("")
    import charts
    figs = []
    for cat, label, n in (("style", "Styles", 8), ("mood", "Mood", 6), ("instruments", "Instruments heard", 6)):
        rows = [(x["tag"][:22], round(100 * x["score"]), f"{x['tag']}: {round(100 * x['score'])}% of your list")
                for x in (t.get(cat) or [])[:n]]
        if rows:
            figs.append(f'<figure>{charts.hbars(rows, width=360, unit="%", label=label, label_w=140, top=100)}'
                        f'<figcaption>{label}: how much of your list each one describes</figcaption></figure>')
    if figs:
        L += [f'<div class="glance">{"".join(figs)}</div>', ""]
    lk = d.get("lesser_known") or []
    if lk:
        L += ["## Lesser-known artists with this sound", "",
              f"_Artists carrying your top tags (**{', '.join(d.get('searched_tags') or [])}**, searched in pairs) on "
              "MusicBrainz that far fewer people play than the artists already in your list (listener counts from "
              "ListenBrainz, the open listening database). Ranked by how many of your tags they share. Artists "
              "already in your list are left out._", "",
              "| Artist | Listeners | Shares your tags | Listen |", "|---|---|---|---|"]
        for a in lk:
            L.append(f"| **{a['name'].replace('|', '/')}** | {a['listeners']:,} | {', '.join(a['shared_tags'])} | "
                     + " · ".join(f"[{site}]({url})" for site, url in a["links"].items()) + " |")
        L.append("")
    elif d.get("searched_tags"):
        L += ["## Lesser-known artists with this sound", "",
              f"We searched for artists tagged {', '.join(d['searched_tags'])} but none were both less played than "
              "your list's artists and listened to by at least a few people. The tag searches below go wider.", ""]
    p = d["search"]
    if p["styles"] or p["moods"]:
        L += ["## Search with these", "",
              "The more specific the tag, the smaller the artists you'll find. Tag pages on Bandcamp, Last.fm and "
              "MusicBrainz list music by tag rather than by play count, and two tags together narrow things down "
              "even more.", ""]
        for s in p["styles"]:
            L.append(f"- **{s['tag']}** ({round(100 * s['score'])}%): " +
                     " · ".join(f"[{site}]({url})" for site, url in s["links"].items()))
        for s in p["moods"]:
            L.append(f"- *{s['tag']}* ({round(100 * s['score'])}%): " +
                     " · ".join(f"[{site}]({url})" for site, url in s["links"].items()))
        if p["combos"]:
            L += ["", "**Try two at once:** " + " · ".join(
                f"{c['query']} ([YouTube]({c['youtube']}), [Spotify]({c['spotify']}))" for c in p["combos"])]
        if p["skip_artists"]:
            L += ["", "Already on your list, so you can skip them in the results: " + ", ".join(p["skip_artists"][:30])]
        L.append("")
    L += ["## Song by song", "", "| Song | Counted as | Top tags |", "|---|---|---|"]
    for s in d["songs"]:
        flat = sorted(((v, k) for cat in s["tags"].values() for k, v in cat.items()), reverse=True)[:6]
        how = (f"listened ({', '.join(s['from'])})" if s["audio"] else
               f"by name ({', '.join(s['from'])})" if s.get("found") else "by name, but we couldn't find it")
        L.append(f"| {s['name'].replace('|', '/')} | {how} | {', '.join(k for _, k in flat) or '—'} |")
    L += ["", "## Where these tags come from", "",
          "- **Our own listening** (the songs you uploaded): we measure the tempo, time signature, key and mode, "
          "energy and mood, whether there are drums, and 8-, 12- and 16-bar blues forms.",
          "- **The style model** (when it's installed): a network trained on a big catalogue of records names "
          "styles from 400 Discogs categories, plus moods, themes and instruments. It's good at broad families "
          "and less sure between close neighbours, like two kinds of techno.",
          "- **Looking up names:** MusicBrainz (and Last.fm, when it's set up) tell us the tags listeners gave "
          "the song or the artist. Songs we only know by name count for less, because an artist's tags describe "
          "their music in general, not necessarily this song.",
          "- Mood words are rules of thumb from music research and from the model's training labels. Take them as "
          "a description, not a fact.", ""]
    if d.get("credit"):
        L += [d["credit"], ""]
    return "\n".join(L)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Tags for finding new music, from songs, a playlist link or a list of names.")
    ap.add_argument("paths", nargs="*", help="audio files (listened to)")
    ap.add_argument("--songs", help="text file of song names, one 'Artist - Title' per line")
    ap.add_argument("--playlist", help="public Spotify playlist/album/track or YouTube playlist link")
    ap.add_argument("--out", default="tags", help="output folder for tags_report.md and tags.json")
    args = ap.parse_args()
    text = Path(args.songs).read_text(encoding="utf-8") if args.songs else None
    if not (args.paths or text or args.playlist):
        ap.error("give some audio files, --songs or --playlist")
    _, report = run_tags(args.paths, args.out, song_text=text, playlist_url=args.playlist, progress=print)
    print(report)


if __name__ == "__main__":
    main()
