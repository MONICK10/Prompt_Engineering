"""
VDOC 2.0 — multi-action command support.

Three stages sit between speech-to-text and the editor:

    splitter        cut one utterance into action chunks
    line_resolver   turn "second line" / "rendavathu vari" into a line target
    sharer          give a chunk the previous chunk's target when it has none
    position_tracker  re-map line numbers after each edit

`pipeline.parse_multi` wires them around the existing DocEditBERT parser.
Nothing here replaces the old single-action path — a command with no safe
split produces exactly one chunk holding the original text, so DocEditBERT
sees a byte-identical input and returns a byte-identical result.
"""
