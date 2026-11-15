"""The theory library: one place for every term, pattern and definition, shared by the professional report (its
underlined words link here) and the educational view (its lessons are built from these entries).

Two kinds of entry:
- lessons (LESSONS below): a pattern or idea worth learning (the 12-bar blues, Dorian, a backbeat), with an
  explanation, how to hear it, well-known recordings that use it, a YouTube search for each and for a lesson,
  and something to try;
- definitions: every word the report explains (musicanalyze.GUIDE and MODE_INFO), so the two never disagree.

Listening examples are the widely documented textbook ones. YouTube links are searches, never guessed video ids,
so they can't point at a wrong or removed video.
"""

import re
from urllib.parse import quote, quote_plus

YOUTUBE = "https://www.youtube.com/results?search_query="
SPOTIFY = "https://open.spotify.com/search/"
SONGSTERR = "https://www.songsterr.com/?pattern="
TRENDS = "https://www.hooktheory.com/trends#"  # key=D&scale=major&path=1.5.6.4 opens D-A-Bm-G (see connect.py)

GROUPS = ["Chords", "Scales & modes", "Progressions", "Form", "Rhythm & groove", "Melody", "Sound & production"]


def slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower().replace("♭", "b").replace("♯", "#").replace("#", "-sharp")).strip("-")


def youtube(query):
    """A YouTube search (not a video id, so it can't go stale or point at the wrong video)."""
    return YOUTUBE + quote_plus(query)


def listen(song, artist):
    """Searches for a listening example: watch it, hear it, play along with its tab."""
    q = f"{song} {artist}"
    return {"url": youtube(q), "spotify": SPOTIFY + quote(q), "songsterr": SONGSTERR + quote_plus(q)}


def L(name, group, short, long, examples=(), lesson=None, try_it=None, related=(), aliases=(), trends=None):
    """trends: (key, Hooktheory scale, path) to open the progression on Hooktheory Trends, in an easy key."""
    return {"name": name, "slug": slug(name), "group": group, "short": short, "long": list(long),
            "trends": f"{TRENDS}key={quote(trends[0])}&scale={trends[1]}&path={trends[2]}" if trends else None,
            "examples": [{"song": s, "artist": a, "note": n, **listen(s, a)} for s, a, n in examples],
            "lesson": youtube(lesson or f"{name} music theory explained"), "try_it": try_it,
            "related": [slug(r) for r in related], "aliases": list(aliases), "kind": "lesson"}


LESSONS = [
    # ------------------------------------------------------------ chords
    L("Major chord", "Chords",
      "Three notes: a root, the major 3rd (4 semitones up) and the 5th (7 up). It sounds bright and settled.",
      ["A major chord stacks the root, the note four semitones above it (the major 3rd) and the note seven "
       "semitones above it (the 5th). C major is C–E–G.",
       "It's written as just the letter: C, G, D. In Roman numerals it's upper case (I, IV, V)."],
      [("Let It Be", "The Beatles", "the whole song is built on four plain chords, three of them major"),
       ("Twist and Shout", "The Beatles", "three major chords, round and round")],
      try_it="Play C–E–G, then lower the E to E♭. The only change is the 3rd, and the mood flips from bright to dark.",
      related=["Minor chord", "Power chord", "Roman numerals"]),
    L("Minor chord", "Chords",
      "A root, the minor 3rd (3 semitones up) and the 5th. It sounds darker or sadder than major.",
      ["The minor chord lowers the major chord's 3rd by a semitone: A minor is A–C–E. That one note carries "
       "most of the emotional difference between major and minor.",
       "It's written with an m (Am, Em). In Roman numerals it's lower case (ii, iii, vi)."],
      [("Stairway to Heaven", "Led Zeppelin", "the intro starts on A minor"),
       ("Zombie", "The Cranberries", "opens on E minor and keeps returning to it")],
      try_it="Play A minor, then C major. They share two notes (C and E): that's why minor and major keys sit so "
             "close together.",
      related=["Major chord", "Relative major and minor", "Aeolian (natural minor)"]),
    L("Power chord", "Chords",
      "Just the root and the 5th (written C5): no 3rd, so it's neither major nor minor.",
      ["Leave the 3rd out of a chord and what's left is open and strong rather than happy or sad. On a "
       "distorted guitar the missing 3rd also stops the chord from turning to mush, which is why rock and metal "
       "live on power chords.",
       "Because it has no 3rd, a power chord can stand in for a major or a minor chord: the melody and the bass "
       "decide which you hear."],
      [("Smells Like Teen Spirit", "Nirvana", "the riff is four power chords"),
       ("You Really Got Me", "The Kinks", "an early, famous power-chord riff"),
       ("Smoke on the Water", "Deep Purple", "the riff is played in parallel 4ths and 5ths")],
      try_it="Play a power chord and sing a major 3rd above it, then a minor 3rd: the same chord fits both.",
      related=["Major chord", "Minor chord", "Riff"], aliases=["C5"]),
    L("Dominant 7th chord", "Chords",
      "A major chord plus the flat 7th (C7 = C–E–G–B♭). Bluesy, and it pulls towards the chord a 5th below.",
      ["Add the note a whole step below the octave to a major chord and you get a dominant 7th. The 3rd and "
       "the 7th form a tense interval (a tritone) that wants to resolve, so G7 leans hard towards C.",
       "In the blues the rule bends: every chord can be a dominant 7th (I7, IV7, V7), and the tension becomes "
       "the colour of the style rather than something to resolve."],
      [("Pride and Joy", "Stevie Ray Vaughan", "a 12-bar blues on dominant 7th chords"),
       ("Purple Haze", "The Jimi Hendrix Experience", "the famous E7♯9: a dominant 7th with an extra crunchy note")],
      lesson="dominant 7th chord explained",
      try_it="Play G7 and stop. Notice how unfinished it sounds, then play C and feel it land.",
      related=["Cadence", "12-bar blues", "ii–V–I (the jazz cadence)"], aliases=["C7"]),
    L("Major 7th chord", "Chords",
      "A major chord plus the major 7th (Cmaj7 = C–E–G–B). Soft, dreamy, jazzy.",
      ["The major 7th sits a semitone below the octave, so it rubs gently against the root. Instead of "
       "pulling anywhere, it makes the chord float: the sound of bossa nova, soul ballads and lo-fi.",
       "Don't confuse it with the dominant 7th (C7), whose 7th is a semitone lower and much more restless."],
      [("Something", "The Beatles", "the opening line moves C to Cmaj7"),
       ("The Girl from Ipanema", "Stan Getz & João Gilberto", "the very first chord is a major 7th")],
      try_it="Play C, then Cmaj7, then C7: three 'C' chords, three different moods.",
      related=["Dominant 7th chord", "Minor 7th chord"], aliases=["Cmaj7"]),
    L("Minor 7th chord", "Chords",
      "A minor chord plus the flat 7th (Am7 = A–C–E–G). Smooth and mellow, less sad than plain minor.",
      ["Adding the 7th softens a minor chord's edges, which is why minor 7ths are everywhere in funk, soul, R&B "
       "and jazz.",
       "In a key, the chord on the 2nd note (ii7) is a minor 7th: it's the first chord of the jazz ii–V–I."],
      [("Get Lucky", "Daft Punk", "the loop starts on a minor 7th"),
       ("So What", "Miles Davis", "long stretches on one minor chord: modal jazz")],
      try_it="Swap every minor chord in a song you know for its minor 7th and listen to it relax.",
      related=["Dorian", "ii–V–I (the jazz cadence)"], aliases=["Cm7"]),
    # ------------------------------------------------------------ scales & modes
    L("Ionian (major)", "Scales & modes",
      "The ordinary major scale (do-re-mi): bright, stable, resolved.",
      ["The major scale's steps go whole, whole, half, whole, whole, whole, half. On a piano, C major is all the "
       "white keys from C.",
       "Ionian is its name as a mode, the starting point the other modes are compared with."],
      [("Let It Be", "The Beatles", "plain major from start to finish"),
       ("Twinkle, Twinkle, Little Star", "traditional", "the first melody many people learn, in major")],
      lesson="major scale explained",
      try_it="Sing do-re-mi-fa-so-la-ti-do. That's Ionian; every other mode changes one or two of these notes.",
      related=["Key", "Aeolian (natural minor)", "Mixolydian", "Lydian"], aliases=["Major scale", "Ionian"]),
    L("Dorian", "Scales & modes",
      "Minor with a raised 6th: soulful, jazzy, funky, less sad than plain minor.",
      ["Take the natural minor scale and raise its 6th note by a semitone. That brighter 6th is Dorian's "
       "colour note: lean on it and the mode comes through.",
       "On a piano, D Dorian is the white keys from D to D."],
      [("Oye Como Va", "Santana", "the two-chord Dorian vamp, i–IV"),
       ("Scarborough Fair", "Simon & Garfunkel", "a folk melody in Dorian"),
       ("So What", "Miles Davis", "modal jazz in D Dorian")],
      lesson="dorian mode explained",
      try_it="Over a held A minor chord, play a melody that uses F♯ (Dorian) and then F (natural minor). Hear the "
             "lift the F♯ brings.",
      related=["Aeolian (natural minor)", "i–IV (Dorian vamp)", "Minor 7th chord"]),
    L("Phrygian", "Scales & modes",
      "Minor with a flat 2nd: dark, tense, a metal and film-score sound.",
      ["Phrygian lowers the 2nd note of the minor scale, so the note just above home is only a semitone away. "
       "That half-step rub gives it its menace.",
       "On a piano, E Phrygian is the white keys from E to E."],
      [("White Rabbit", "Jefferson Airplane", "a bolero-like build with a Phrygian, Spanish flavour"),
       ("Wherever I May Roam", "Metallica", "a riff leaning on the flat 2nd")],
      lesson="phrygian mode explained",
      try_it="Play E and F together, then alone, one after the other. That half step is Phrygian in a nutshell.",
      related=["Phrygian dominant", "Aeolian (natural minor)"]),
    L("Lydian", "Scales & modes",
      "Major with a raised 4th: dreamy, floating, film-score.",
      ["Raise the 4th note of the major scale and the music seems to lift off the ground: the raised 4th "
       "doesn't pull back to the 3rd the way the plain 4th does.",
       "On a piano, F Lydian is the white keys from F to F."],
      [("The Simpsons theme", "Danny Elfman", "the opening melody leans on the raised 4th"),
       ("Flying in a Blue Dream", "Joe Satriani", "a guitar piece built on Lydian")],
      lesson="lydian mode explained",
      try_it="Over a C chord, play a melody that uses F♯ instead of F. That's the Lydian 'float'.",
      related=["Ionian (major)", "Mixolydian"]),
    L("Mixolydian", "Scales & modes",
      "Major with a flat 7th: bluesy, rock, folk, laid-back.",
      ["Lower the 7th note of the major scale and the music loses its urge to resolve upwards: it relaxes, "
       "the way rock, folk and blues do. The chord built on that flat 7th (♭VII) is a rock favourite.",
       "On a piano, G Mixolydian is the white keys from G to G."],
      [("Norwegian Wood", "The Beatles", "a folk melody in Mixolydian"),
       ("Sweet Home Alabama", "Lynyrd Skynyrd", "D–C–G: the ♭VII chord, rock style")],
      lesson="mixolydian mode explained",
      try_it="Play G–F–C–G (I–♭VII–IV–I). That F chord, foreign to G major, is the Mixolydian sound.",
      related=["I–♭VII–IV (Mixolydian rock)", "Ionian (major)", "Dominant 7th chord"]),
    L("Aeolian (natural minor)", "Scales & modes",
      "The ordinary minor scale: sad, serious, emotional.",
      ["The natural minor scale steps go whole, half, whole, whole, half, whole, whole. A minor is the white keys "
       "from A, which is why A minor and C major share all their notes.",
       "Aeolian is its name as a mode."],
      [("Stairway to Heaven", "Led Zeppelin", "the intro is in A minor"),
       ("Losing My Religion", "R.E.M.", "a minor-key song built round A minor"),
       ("Rolling in the Deep", "Adele", "a minor-key pop hit")],
      lesson="natural minor scale explained",
      try_it="Play the white keys from A to A, then from C to C: same notes, different home, different mood.",
      related=["Minor chord", "Relative major and minor", "Dorian", "Harmonic minor"],
      aliases=["Natural minor", "Minor scale"]),
    L("Locrian", "Scales & modes",
      "Flat 2nd and flat 5th: so unstable it hardly ever works as a home key.",
      ["Locrian's home chord is diminished (its 5th is flattened), so the music never feels settled. It's "
       "mostly used for brief moments of unease rather than whole songs.",
       "On a piano, B Locrian is the white keys from B to B."],
      [("Army of Me", "Björk", "often given as a rare example of a Locrian-flavoured riff")],
      lesson="locrian mode explained",
      try_it="Play the white keys from B to B and try to make B feel like home. It resists: that's Locrian.",
      related=["Phrygian"]),
    L("Harmonic minor", "Scales & modes",
      "Minor with a raised 7th: dramatic, classical, with an exotic pull to home.",
      ["Raising the minor scale's 7th gives a leading note a semitone below home, so the V chord becomes "
       "major and pulls strongly to i. The gap between the 6th and the raised 7th (three semitones) gives it its "
       "'exotic' sound.",
       "It's the minor of Baroque and Classical music and of neoclassical metal."],
      [("Bourrée in E minor", "J. S. Bach", "minor-key Baroque writing"),
       ("Far Beyond the Sun", "Yngwie Malmsteen", "neoclassical metal in harmonic minor")],
      lesson="harmonic minor scale explained",
      try_it="Play A minor, then E major, then A minor again. The G♯ in E major is the harmonic minor's raised 7th.",
      related=["Aeolian (natural minor)", "Phrygian dominant", "Cadence"]),
    L("Phrygian dominant", "Scales & modes",
      "A flat 2nd with a major 3rd: Spanish, flamenco, Middle Eastern.",
      ["This is the 5th mode of harmonic minor: Phrygian's dark flat 2nd, but with a major 3rd. The 1½-step gap "
       "between them is the sound people call 'Spanish' or 'Middle Eastern'."],
      [("Misirlou", "Dick Dale", "surf guitar on Phrygian dominant"),
       ("Hava Nagila", "traditional", "the classic example")],
      lesson="phrygian dominant scale explained",
      try_it="Play E–F–G♯–A–B–C–D–E slowly and listen to the jump from F to G♯.",
      related=["Phrygian", "Harmonic minor"]),
    L("Pentatonic scale", "Scales & modes",
      "A five-note scale with no semitones, so almost nothing in it can clash.",
      ["The major pentatonic is the major scale without its 4th and 7th (C–D–E–G–A); the minor pentatonic is "
       "the minor scale without its 2nd and 6th (A–C–D–E–G). Same five notes, different home.",
       "It's the first scale most guitarists learn to improvise with, and it's found in folk music all over "
       "the world."],
      [("My Girl", "The Temptations", "the guitar riff is major pentatonic"),
       ("Amazing Grace", "traditional", "a pentatonic melody"),
       ("Stairway to Heaven", "Led Zeppelin", "the guitar solo is mostly A minor pentatonic")],
      lesson="pentatonic scale explained",
      try_it="On a piano, improvise using only the black keys: that's a pentatonic scale, and it always sounds fine.",
      related=["Blues scale", "Ionian (major)", "Aeolian (natural minor)"]),
    L("Blues scale", "Scales & modes",
      "The minor pentatonic plus one 'blue note', the flat 5th.",
      ["A–C–D–E♭–E–G: the minor pentatonic with the flat 5th squeezed in. Blues players also bend the minor 3rd "
       "towards the major 3rd, so the notes between the keys become part of the sound.",
       "Played over major chords it gives the blues its mix of bright and sad."],
      [("Sunshine of Your Love", "Cream", "a riff on the blues scale"),
       ("Smoke on the Water", "Deep Purple", "the riff includes the blue note")],
      lesson="blues scale explained",
      try_it="Over an A7 chord, slide from C to C♯ (minor 3rd to major 3rd). That smear is the blues.",
      related=["Pentatonic scale", "12-bar blues", "Major/minor blur"]),
    # ------------------------------------------------------------ progressions
    L("I–V–vi–IV (the 'four chords' of pop)", "Progressions",
      "The most-used chord loop in modern pop: home, the 5th, the relative minor, the 4th.",
      ["In C major that's C–G–Am–F. It balances bright and dark (three major chords, one minor) and loops back "
       "to the start smoothly, which is why so many songs share it.",
       "Start the same loop on the minor chord (vi–IV–I–V) and it sounds moodier, though it's the same four chords."],
      [("Let It Be", "The Beatles", "the verse"),
       ("No Woman, No Cry", "Bob Marley", "the loop under the whole song"),
       ("With or Without You", "U2", "the loop under the whole song"),
       ("Someone Like You", "Adele", "the loop on piano"),
       ("Don't Stop Believin'", "Journey", "the verse"),
       ("Zombie", "The Cranberries", "as vi–IV–I–V")],
      lesson="I V vi IV four chord song explained", trends=("C", "major", "1.5.6.4"),
      try_it="Play C–G–Am–F, then start it on Am (Am–F–C–G). Same chords: which one sounds sadder?",
      related=["Roman numerals", "I–vi–IV–V (the '50s progression)", "Relative major and minor"],
      aliases=["I–V–vi–IV"]),
    L("I–vi–IV–V (the '50s progression)", "Progressions",
      "Home, the relative minor, the 4th, the 5th: the doo-wop sound.",
      ["In C that's C–Am–F–G. The V chord at the end pulls straight back to I, so the loop feels like a circle. "
       "It defined late-1950s doo-wop and still turns up in ballads."],
      [("Stand by Me", "Ben E. King", "the loop under the whole song"),
       ("Every Breath You Take", "The Police", "the verse"),
       ("Earth Angel", "The Penguins", "classic doo-wop")],
      lesson="50s doo wop progression I vi IV V explained", trends=("C", "major", "1.6.4.5"),
      try_it="Play C–Am–F–G slowly with a 'shoo-bop' rhythm and you're in 1958.",
      related=["I–V–vi–IV (the 'four chords' of pop)", "Cadence"], aliases=["I–vi–IV–V"]),
    L("Andalusian cadence", "Progressions",
      "Four chords stepping down from the minor home chord: i–♭VII–♭VI–V.",
      ["In A minor that's Am–G–F–E. The bass walks down step by step and the major V chord at the end (with its "
       "G♯) pulls back to the start. It's flamenco's signature and a pop favourite since the '60s."],
      [("Hit the Road Jack", "Ray Charles", "the loop under the whole song"),
       ("Runaway", "Del Shannon", "the verse"),
       ("Walk, Don't Run", "The Ventures", "the main theme"),
       ("Stray Cat Strut", "Stray Cats", "the verse")],
      lesson="andalusian cadence explained",
      try_it="Play Am–G–F–E with a strong down-strum and listen to the bass walk down A–G–F–E.",
      related=["Aeolian (natural minor)", "Harmonic minor", "i–♭VII–♭VI (descending minor)"],
      aliases=["i–bVII–bVI–V (Andalusian cadence)"]),
    L("I–♭VII–IV (Mixolydian rock)", "Progressions",
      "Home, the chord a whole step below, then the 4th: the classic rock move.",
      ["In D that's D–C–G. The C chord isn't in D major; it comes from the Mixolydian mode, and it gives rock "
       "its swagger. The 'na na na' coda of Hey Jude uses the same idea (F–E♭–B♭–F)."],
      [("Sweet Child o' Mine", "Guns N' Roses", "the verse chords"),
       ("Sympathy for the Devil", "The Rolling Stones", "the loop under the song"),
       ("Hey Jude", "The Beatles", "the long coda"),
       ("Sweet Home Alabama", "Lynyrd Skynyrd", "D–C–G round and round")],
      lesson="bVII chord rock progression explained", trends=("C", "major", "1.b7.4"),
      try_it="Play D–C–G–D with a strong backbeat: instant classic rock.",
      related=["Mixolydian", "I–IV–V (three-chord rock'n'roll)"], aliases=["I–bVII–IV (Mixolydian rock)"]),
    L("i–♭VII–♭VI (descending minor)", "Progressions",
      "The minor home chord stepping down twice: dark, driving, epic.",
      ["In C♯ minor that's C♯m–B–A. The chords walk down by step and loop back up, which gives an urgent, "
       "circling feeling."],
      [("All Along the Watchtower", "Bob Dylan / Jimi Hendrix", "the loop under the whole song")],
      lesson="i bVII bVI progression explained", trends=("A", "minor", "1.7.6"),
      try_it="Loop Am–G–F–G and sing over it: most minor-key rock is hiding in there.",
      related=["Andalusian cadence", "Aeolian (natural minor)"], aliases=["i–bVII–bVI (descending minor)"]),
    L("I–IV–V (three-chord rock'n'roll)", "Progressions",
      "The three main chords of any major key: the bones of rock'n'roll, folk and country.",
      ["I is home, IV moves away, and V pulls back home. In A that's A–D–E. With these three chords you can "
       "play thousands of songs, and the 12-bar blues is the same three chords in a fixed order."],
      [("La Bamba", "Ritchie Valens", "I–IV–V on repeat"),
       ("Twist and Shout", "The Isley Brothers / The Beatles", "I–IV–V on repeat"),
       ("Wild Thing", "The Troggs", "A–D–E")],
      lesson="I IV V chord progression explained", trends=("C", "major", "1.4.5"),
      try_it="Learn the I, IV and V chords in G (G, C, D) and play any campfire song you know.",
      related=["12-bar blues", "Roman numerals", "Cadence"], aliases=["I–IV–V"]),
    L("i–IV (Dorian vamp)", "Progressions",
      "Two chords back and forth: a minor home chord and a major chord on the 4th.",
      ["In A minor the IV chord would normally be D minor; making it D major borrows Dorian's raised 6th (F♯). "
       "Vamped over and over, it's the sound of Latin rock and funk."],
      [("Oye Como Va", "Santana", "Am7–D9 for the whole song"),
       ("Evil Ways", "Santana", "Gm–C on repeat")],
      lesson="dorian vamp i IV explained", trends=("A", "dorian", "1.4"),
      try_it="Loop Am–D and improvise with the A Dorian scale (the white keys from G, starting on A).",
      related=["Dorian"], aliases=["i–IV"]),
    L("ii–V–I (the jazz cadence)", "Progressions",
      "Minor 7th on the 2nd, dominant 7th on the 5th, then home: the backbone of jazz.",
      ["In C that's Dm7–G7–Cmaj7. Each chord's root is a 5th above the next one's, so the harmony falls home "
       "naturally. Jazz standards chain ii–V–Is through many keys."],
      [("Autumn Leaves", "jazz standard", "a chain of ii–V–Is"),
       ("Fly Me to the Moon", "Frank Sinatra", "ii–V–Is throughout"),
       ("Honeysuckle Rose", "Fats Waller", "opens with a ii–V")],
      lesson="ii V I jazz progression explained", trends=("C", "major", "2.5.1"),
      try_it="Play Dm7–G7–Cmaj7, then the same shape a whole step down (Cm7–F7–B♭maj7).",
      related=["Minor 7th chord", "Dominant 7th chord", "Cadence"], aliases=["ii–V–I"]),
    L("Cadence", "Progressions",
      "The chords that end a phrase, like punctuation: a full stop, a comma, a surprise.",
      ["A perfect (authentic) cadence goes V–I: the full stop. A plagal cadence goes IV–I, the 'Amen' at the end "
       "of hymns. A half cadence stops on V, a comma. A deceptive cadence goes V–vi: you expect home and get the "
       "minor chord instead."],
      [("Amen (the end of a hymn)", "traditional", "the plagal IV–I cadence"),
       ("Twinkle, Twinkle, Little Star", "traditional", "each phrase ends with a V–I")],
      lesson="cadences explained perfect plagal deceptive",
      try_it="Play G7–C (perfect), then G7–Am (deceptive). Feel the surprise.",
      related=["Dominant 7th chord", "Roman numerals", "Turnaround"]),
    L("Turnaround", "Progressions",
      "A short run of chords at the end of a section that leads back to the start.",
      ["In a 12-bar blues the last two bars (often I–V) turn the music around for the next chorus. In jazz, "
       "turnarounds such as I–vi–ii–V fill the gap before the top of the form."],
      [("Sweet Home Chicago", "Robert Johnson", "a classic blues turnaround lick in the last bars")],
      lesson="blues turnaround explained",
      try_it="End a 12-bar blues on V instead of I and hear how it pulls you into the next chorus.",
      related=["12-bar blues", "Cadence"]),
    # ------------------------------------------------------------ form
    L("12-bar blues", "Form",
      "Twelve bars, three chords in a fixed order: I for 4 bars, IV for 2, I for 2, then V, IV, I, I.",
      ["The 12-bar blues is a form, not just a sound: three lines of four bars. The first line sits on the home "
       "chord (I), the second moves to the IV chord and back, and the third goes V–IV–I and turns around.",
       "One pass through the twelve bars is a 'chorus'. Verses, solos and endings all follow the same chorus, so "
       "musicians who have never met can play a blues together.",
       "Variations: the 'quick change' goes to IV in bar 2, a turnaround puts V in bar 12, and jazz blues add "
       "ii–V moves."],
      [("Sweet Home Chicago", "Robert Johnson", "Delta and Chicago blues"),
       ("Pride and Joy", "Stevie Ray Vaughan", "Texas shuffle"),
       ("Johnny B. Goode", "Chuck Berry", "rock'n'roll is a 12-bar blues sped up"),
       ("Hound Dog", "Big Mama Thornton", "the original 1952 recording"),
       ("Tutti Frutti", "Little Richard", "rock'n'roll on the 12-bar"),
       ("Billie's Bounce", "Charlie Parker", "a bebop jazz blues")],
      lesson="12 bar blues explained",
      try_it="Count to four twelve times while changing chords: A A A A | D D A A | E D A A. That's the form.",
      related=["I–IV–V (three-chord rock'n'roll)", "Dominant 7th chord", "Blues scale", "Turnaround",
               "Quick change", "8-bar blues"]),
    L("8-bar blues", "Form",
      "A shorter blues: eight bars that turn round faster, often I–V–IV–IV–I–V–I–V.",
      ["Four bars shorter than the 12-bar, the 8-bar blues feels quicker and more song-like. There's no single "
       "fixed version, but it always cycles the I, IV and V chords."],
      [("Key to the Highway", "Big Bill Broonzy", "the classic 8-bar"),
       ("Heartbreak Hotel", "Elvis Presley", "an 8-bar blues hit")],
      lesson="8 bar blues explained",
      related=["12-bar blues", "16-bar blues"]),
    L("16-bar blues", "Form",
      "The 12-bar idea stretched to sixteen bars, often with eight bars on the home chord.",
      ["Some 16-bar blues stay on I for eight bars before the change; others extend the ending with a longer "
       "V–IV turnaround."],
      [("Hoochie Coochie Man", "Muddy Waters", "eight bars on I before the change")],
      lesson="16 bar blues explained", related=["12-bar blues", "8-bar blues"]),
    L("Quick change", "Form",
      "A 12-bar blues that visits the IV chord in bar 2, then comes straight back.",
      ["Instead of four bars on I, the quick change goes I–IV–I–I. It breaks up the long opening on the home "
       "chord and is common in Chicago and Texas blues."],
      lesson="quick change blues explained", related=["12-bar blues"]),
    L("AABA", "Form",
      "The 32-bar song form: an 8-bar A played twice, a contrasting 8-bar bridge (B), then A again.",
      ["It was the standard shape of Tin Pan Alley songs and jazz standards: the A tune is the hook, and the "
       "bridge (the 'middle eight') gives the ear something new before the hook returns.",
       "Jazz players call one pass through all 32 bars a 'chorus', and solo over it again and again."],
      [("Over the Rainbow", "Judy Garland", "the textbook AABA"),
       ("Yesterday", "The Beatles", "a pop song in AABA form"),
       ("I Got Rhythm", "George Gershwin", "the 'rhythm changes' every jazz player knows"),
       ("Blue Moon", "Rodgers & Hart", "a 32-bar standard")],
      lesson="AABA song form explained",
      try_it="Sing a tune you know and mark where it changes for 8 bars and then comes back: that's the B.",
      related=["Verse–chorus form", "Form"], aliases=["AABA (32-bar song form)"]),
    L("Verse–chorus form", "Form",
      "Verses tell the story with new words each time; the chorus repeats the same words and the hook.",
      ["Most pop and rock is verse–chorus: verse, chorus, verse, chorus, bridge, chorus. The chorus is usually "
       "the loudest, highest and most repeated part, and the bridge is a one-time detour."],
      [("Smells Like Teen Spirit", "Nirvana", "quiet verses, loud choruses"),
       ("Billie Jean", "Michael Jackson", "verse, pre-chorus, chorus")],
      lesson="verse chorus song structure explained",
      try_it="Map a song you love with letters: A for each verse, B for each chorus. Where's the bridge?",
      related=["AABA", "Form"]),
    # ------------------------------------------------------------ rhythm
    L("Backbeat", "Rhythm & groove",
      "The snare on beats 2 and 4: the heartbeat of rock, pop, soul and R&B.",
      ["Count '1 2 3 4': the kick drum tends to land on 1 and 3, the snare (or a clap) on 2 and 4. Accenting the "
       "'off' beats is what makes people clap and dance on 2 and 4."],
      [("Billie Jean", "Michael Jackson", "a crisp, steady backbeat"),
       ("We Will Rock You", "Queen", "stomp, stomp, CLAP: the clap is the backbeat")],
      lesson="backbeat drum beat explained",
      try_it="Tap your foot on every beat and clap only on 2 and 4.",
      related=["Four-on-the-floor", "Half-time feel", "Syncopation"]),
    L("Four-on-the-floor", "Rhythm & groove",
      "A kick drum on every beat: the disco and house pulse.",
      ["Four even kicks a bar, usually with an off-beat hi-hat ('tss') between them. At 115–130 BPM it's the "
       "engine of disco, house and techno."],
      [("Stayin' Alive", "Bee Gees", "disco"),
       ("One More Time", "Daft Punk", "French house"),
       ("I Feel Love", "Donna Summer", "early electronic disco")],
      lesson="four on the floor beat explained",
      try_it="Stomp on every beat and say 'tss' between the stomps.",
      related=["Backbeat", "Sidechain pumping"]),
    L("Half-time feel", "Rhythm & groove",
      "The snare only on beat 3, so the groove feels half as fast as the tempo.",
      ["The hi-hats can keep going at the full tempo while the snare lands once a bar. Heavy rock choruses, "
       "dubstep and trap beats use it to make things feel huge and slow."],
      lesson="half time feel drums explained", related=["Backbeat", "Tempo"]),
    L("Tresillo (3+3+2)", "Rhythm & groove",
      "Eight pulses grouped 3+3+2: the rhythm under reggaeton, dancehall and a lot of pop.",
      ["Count eight quick pulses and accent 1, 4 and 7. It comes from Afro-Cuban music and runs through Latin, "
       "Caribbean and modern pop."],
      [("Shape of You", "Ed Sheeran", "the marimba riff and beat"),
       ("Despacito", "Luis Fonsi ft. Daddy Yankee", "the reggaeton dembow beat")],
      lesson="tresillo 3 3 2 rhythm explained",
      try_it="Count 1-2-3-4-5-6-7-8 and clap on 1, 4 and 7.",
      related=["Syncopation"], aliases=["3+3+2"]),
    L("Syncopation", "Rhythm & groove",
      "Accents between the beats instead of on them: what makes a groove feel pushed or funky.",
      ["When a note lands just before a beat, or on the 'and', the ear expects one thing and gets another. That "
       "surprise is funk, reggae, Latin music and most dance grooves."],
      [("Superstition", "Stevie Wonder", "the clavinet riff"),
       ("Take On Me", "a-ha", "the synth riff pushes ahead of the beat")],
      lesson="syncopation explained",
      try_it="Clap on the beats of a song, then clap only on the 'and's in between.",
      related=["Backbeat", "Swing", "Tresillo (3+3+2)"]),
    L("Swing", "Rhythm & groove",
      "Pairs of notes played long–short instead of even: the lilt of jazz, blues shuffles and swing.",
      ["In swing, the first of two eighth notes is held longer and the second is shorter, roughly a 2:1 split. "
       "A blues 'shuffle' is the same feel."],
      [("In the Mood", "Glenn Miller", "big-band swing"),
       ("Take the 'A' Train", "Duke Ellington", "swing era classic"),
       ("Pride and Joy", "Stevie Ray Vaughan", "a Texas shuffle")],
      lesson="swing rhythm explained", related=["Syncopation", "12-bar blues"], aliases=["Shuffle"]),
    L("Waltz time (3/4)", "Rhythm & groove",
      "Three beats a bar, ONE-two-three: the sway of waltzes and many ballads.",
      ["Strong, weak, weak. It feels like a circle or a sway rather than a march."],
      [("The Blue Danube", "Johann Strauss II", "the famous waltz"),
       ("Piano Man", "Billy Joel", "a pop song in 3/4"),
       ("Moon River", "Henry Mancini", "a waltz ballad")],
      lesson="3/4 time signature explained",
      try_it="Count ONE-two-three and sway on the ONE.",
      related=["Meter", "Odd meters (5/4, 7/4)"], aliases=["3/4"]),
    L("Odd meters (5/4, 7/4)", "Rhythm & groove",
      "Bars of five or seven beats: they limp or lurch in an exciting way.",
      ["A 5/4 bar is usually felt as 3+2 or 2+3, a 7/4 bar as 4+3. Prog rock, jazz and film music use them to "
       "keep the listener slightly off balance."],
      [("Take Five", "The Dave Brubeck Quartet", "jazz in 5/4"),
       ("Mission: Impossible theme", "Lalo Schifrin", "5/4"),
       ("Money", "Pink Floyd", "7/4")],
      lesson="odd time signatures 5/4 7/4 explained",
      try_it="Count 1-2-3-1-2 over and over: that's 5/4.",
      related=["Meter", "Waltz time (3/4)"], aliases=["5/4", "7/4"]),
    # ------------------------------------------------------------ melody
    L("Riff", "Melody",
      "A short, catchy figure repeated over and over, often in the guitar or bass.",
      ["A riff is the hook you can hum without words. It usually repeats every bar or two and carries the song's "
       "identity."],
      [("Smoke on the Water", "Deep Purple", "the most famous guitar riff"),
       ("Seven Nation Army", "The White Stripes", "a riff everyone sings"),
       ("(I Can't Get No) Satisfaction", "The Rolling Stones", "a fuzz guitar riff")],
      lesson="what is a guitar riff",
      try_it="Make up a 4-note riff and repeat it 8 times without changing it, then change one note.",
      related=["Motif", "Pentatonic scale"]),
    L("Motif", "Melody",
      "A short musical idea that keeps coming back, changed a little each time.",
      ["Composers build whole pieces from a motif: repeating it, moving it higher, turning it upside down. Even "
       "two notes can be one."],
      [("Symphony No. 5", "Ludwig van Beethoven", "short-short-short-long"),
       ("Jaws theme", "John Williams", "two notes, a semitone apart")],
      lesson="musical motif explained",
      try_it="Take a 3-note idea and play it, then a step higher, then backwards.",
      related=["Riff", "Call and response"]),
    L("Call and response", "Melody",
      "One phrase asks, another answers: a voice and a choir, a singer and a guitar.",
      ["It comes from African and African-American music: work songs, gospel and the blues, where the guitar "
       "answers each sung line."],
      [("Hit the Road Jack", "Ray Charles", "Ray and the Raelettes trade lines"),
       ("Shout", "The Isley Brothers", "the singer calls, the band answers")],
      lesson="call and response music explained",
      try_it="Sing a short phrase, then answer it with one that ends on the home note.",
      related=["Motif", "12-bar blues"]),
    # ------------------------------------------------------------ harmony basics with lessons
    L("Key", "Scales & modes",
      "The home note and scale a piece keeps coming back to, like E major or A minor.",
      ["The key tells you which notes belong and which chord feels like home. Most songs stay in one key; some "
       "move to another (a modulation) for a lift."],
      lesson="what is a musical key explained",
      try_it="Hum the last note of a song you know. That's almost always the key's home note.",
      related=["Relative major and minor", "Modulation", "Ionian (major)"]),
    L("Relative major and minor", "Scales & modes",
      "A major key and a minor key that share all their notes, like C major and A minor.",
      ["Every major key has a relative minor three semitones below its home note. Same notes, different home, "
       "different mood, which is why the two are easy to mix up and easy to move between."],
      lesson="relative major and minor explained",
      try_it="Play C–G–Am–F, then Am–F–C–G: same chords, but the second feels like A minor.",
      related=["Key", "Aeolian (natural minor)", "Ionian (major)"]),
    L("Modulation", "Scales & modes",
      "Moving to a new key partway through a song.",
      ["A step up for the last chorus (the 'truck driver's gear change') lifts the energy; a move to a distant key "
       "can make a bridge feel like another world."],
      [("Man in the Mirror", "Michael Jackson", "the key goes up for the last choruses"),
       ("Love on Top", "Beyoncé", "several key changes in a row"),
       ("My Girl", "The Temptations", "the key rises partway through")],
      lesson="key change modulation explained",
      try_it="Sing a chorus, then sing it again a whole step higher.",
      related=["Key"], aliases=["Key change"]),
    L("Roman numerals", "Progressions",
      "Chords named by where they sit in the key, so a progression can be described in any key.",
      ["I is the home chord, IV and V are built on the 4th and 5th notes. Upper case is major, lower case "
       "minor: in C major, I = C, ii = Dm, iii = Em, IV = F, V = G, vi = Am."],
      lesson="roman numeral chords explained",
      try_it="Write the I, IV, V and vi chords of G major (G, C, D, Em) and play I–V–vi–IV.",
      related=["Key", "I–V–vi–IV (the 'four chords' of pop)", "I–IV–V (three-chord rock'n'roll)"]),
    L("Pedal / drone", "Melody",
      "One note held or repeated while the chords change above it.",
      ["A drone grounds the music and makes the changing chords rub against it. Bagpipes, the Indian tanpura and "
       "a bass holding one note under a whole verse all use it."],
      [("Within You Without You", "The Beatles", "over an Indian-style drone")],
      lesson="pedal point drone music explained",
      try_it="Hold a low D and play D, C, G and D chords above it.",
      related=["Key"], aliases=["Drone / pedal"]),
    L("Major/minor blur", "Scales & modes",
      "Both the major and the minor 3rd sound: the 'blue note' that makes blues and rock sound the way they do.",
      ["Over a major chord, a blues player bends or slides the minor 3rd up towards the major 3rd. Neither "
       "fully major nor minor, it's the blues' signature."],
      lesson="blue notes explained", related=["Blues scale", "12-bar blues"]),
    # ------------------------------------------------------------ production
    L("Sidechain pumping", "Sound & production",
      "Pads and synths duck right after each kick and swell back: the 'breathing' of dance music.",
      ["A compressor on the synths listens to the kick drum and turns them down for a moment each time it hits. "
       "Heavy pumping (6 dB or more) became the sound of French house."],
      [("Call on Me", "Eric Prydz", "obvious, heavy pumping"),
       ("One More Time", "Daft Punk", "French house pumping")],
      lesson="sidechain compression explained",
      related=["Four-on-the-floor", "Drop"], aliases=["Sidechain"]),
    L("Drop", "Sound & production",
      "In dance music, the moment the full beat and bass come back after a build or breakdown.",
      ["A build raises the tension (rising noise, a snare roll, the bass taken out); the drop releases it all at "
       "once. Arranging in 8-, 16- and 32-bar blocks makes the drop land where dancers expect it."],
      [("Levels", "Avicii", "builds and drops")],
      lesson="EDM build up and drop explained", related=["Sidechain pumping"]),
]

BY_SLUG = {e["slug"]: e for e in LESSONS}
assert len(BY_SLUG) == len(LESSONS), "two lessons share a slug"


def _definitions():
    """Every word the report explains, so the library and the report's tooltips never disagree."""
    import musicanalyze as ma
    out = {}
    for group, terms in ma.GUIDE.items():
        for term, text in terms.items():
            out[term] = {"name": term, "slug": slug(term), "group": group, "short": text, "long": [],
                         "examples": [], "lesson": youtube(f"{term} music explained"), "try_it": None, "trends": None,
                         "related": [], "aliases": [], "kind": "definition"}
    for mode, text in ma.MODE_INFO.items():
        out.setdefault(mode, {"name": mode, "slug": slug(mode), "group": "Scales & modes", "short": text, "long": [],
                              "examples": [], "lesson": youtube(f"{mode} mode explained"), "try_it": None,
                              "trends": None,
                              "related": [], "aliases": [], "kind": "definition"})
    return out


_alias = None


def _aliases():
    global _alias
    if _alias is None:
        _alias = {}
        for e in LESSONS:
            for a in [e["name"]] + e["aliases"]:
                _alias[a.lower()] = e["slug"]
                _alias[slug(a)] = e["slug"]
    return _alias


def find(term):
    """The library entry for a word or name (a lesson first, else the report's definition), or None."""
    if not term:
        return None
    s = _aliases().get(term.lower()) or _aliases().get(slug(term))
    if s:
        e = dict(BY_SLUG[s])
        d = _definitions().get(term)  # the report's own wording, when it explains this term too
        if d and d["short"] != e["short"]:
            e["in_report"] = d["short"]
        return e
    d = _definitions().get(term)
    if d:
        return d
    d = next((x for x in _definitions().values() if x["slug"] == slug(term)), None)
    if d is None and " (" in term:  # "I–vi–IV–V (the '50s / doo-wop progression)": the name without its nickname
        return find(term.split(" (")[0])
    return d


def entries():
    """Everything in the library: the lessons, then each report definition that has no lesson of its own."""
    out = [dict(e) for e in LESSONS]
    for term, d in _definitions().items():
        s = _aliases().get(term.lower()) or _aliases().get(slug(term))
        if s:
            next(e for e in out if e["slug"] == s).setdefault("in_report", d["short"])
        elif not any(e["slug"] == d["slug"] for e in out):
            out.append(d)
    return out


def get(s):
    return next((e for e in entries() if e["slug"] == s), None)


_index = None


def index():
    """{a term's slug, or a name or alias in lower case: the library slug}: lets a report's underlined words (old
    reports by their text, new ones by their data-term) link to the library."""
    global _index
    if _index is None:
        _index = {}
        for e in entries():
            for k in [e["slug"], e["name"].lower()] + [a.lower() for a in e["aliases"]]:
                _index.setdefault(k, e["slug"])
        for term in _definitions():  # a report term that has a lesson goes to the lesson
            f = find(term)
            _index[slug(term)] = _index[term.lower()] = f["slug"]
    return _index


def grouped(q=""):
    """{group: [entries]} in the library's order, filtered by a search (name, aliases or text)."""
    q = (q or "").strip().lower()
    out = {}
    for e in entries():
        text = " ".join([e["name"], e["short"], " ".join(e["long"]), " ".join(e["aliases"]),
                         " ".join(x["song"] + " " + x["artist"] for x in e["examples"])]).lower()
        if q and q not in text:
            continue
        out.setdefault(e["group"], []).append(e)
    order = GROUPS + [g for g in out if g not in GROUPS]
    return {g: sorted(out[g], key=lambda e: (e["kind"] != "lesson", e["name"].lower())) for g in order if g in out}
