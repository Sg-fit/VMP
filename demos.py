"""Demo results: a full report to look at before uploading anything (remusic.ai's sample tracks do the same).

The demos are made from generated audio - the same synthesizers the self-test uses - so there are no rights
issues and the content is known. They sound simpler than a real recording, and the page says so. Each demo is
made once, the first time someone opens it, then kept: it's a normal job folder whose id depends on the code
version, so a new release makes fresh demos and the old ones expire like any other result.
"""

import hashlib
from pathlib import Path

VERSION = 1  # bump when the demo material changes

DEMOS = {
    "blues": {
        "title": "Solo guitar blues",
        "about": "A 12-bar blues on one guitar, no drums or bass: the key and form, each chorus, the guitar's own "
                 "low notes as sheet music, and blues ideas.",
    },
    "band": {
        "title": "A band song, next to a reference",
        "about": "Drums, bass, chords and a typed melody, measured beside a faster, louder song: what that song does "
                 "differently, and what to try.",
    },
    "electronic": {
        "title": "House track",
        "about": "Four on the floor at 124 BPM: the drum grid, the synth bass, the energy arc and loudness for "
                 "streaming.",
    },
}


def job_id(slug, code_version):
    """A fixed 32-hex job id per demo and code version (a new release regenerates the demos)."""
    return hashlib.md5(f"demo|{slug}|{code_version}|{VERSION}".encode()).hexdigest()


def prepare(slug, d):
    """Write the demo's generated audio into job folder `d`; return the status fields for the job."""
    import musicanalyze as ma
    import selftest as st
    d = Path(d)
    (d / "input").mkdir(parents=True, exist_ok=True)
    fields = {"mode": "musician", "demo": slug, "use_ai": False, "melody": "", "context": None, "reference": None}
    if slug == "blues":
        name = "Solo guitar blues (generated).wav"
        st.write(d / "input" / name, st.guitar_blues())
        fields["context"] = ma.clean_context(["guitar"], "blues")
    elif slug == "band":
        name = "My pop song (generated).wav"
        G4, A4, B4, D5 = st.midi("G", 4), st.midi("A", 4), st.midi("B", 4), st.midi("D", 5)
        melody = [[B4, A4, G4, D5], [A4, B4, D5, A4], [G4, B4, A4, G4], [D5, B4, A4, G4]]
        st.write(d / "input" / name, st.song([("G", ""), ("D", ""), ("E", "m"), ("C", "")], 100, 4, loops=8,
                                              melody=melody))
        fields["melody"] = "B4 A4 G4 D5 A4 B4 D5 A4"
        ref = "A song I admire (generated).wav"
        (d / "reference").mkdir(exist_ok=True)
        st.write(d / "reference" / ref, st.song([("E", "m"), ("C", ""), ("G", ""), ("D", "")], 118, 4, loops=10))
        fields["reference"] = ref
    elif slug == "electronic":
        name = "House track (generated).wav"
        st.write(d / "input" / name, st.electronic_track())
    else:
        raise KeyError(slug)
    fields["files"] = [name]
    return fields
