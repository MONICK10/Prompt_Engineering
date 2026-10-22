"""
Position tracker.

Line numbers are not stable. After

    "second line delete pannu, apram second line bold pannu"

the second "second line" must mean the line that *moved up* into slot 2, not
the line that was in slot 2 when the sentence started.

The tracker keeps a list of stable line ids. Resolving a line number is a
lookup into that list as it stands right now; deleting a line removes its id,
so every later lookup sees the shifted document. Anchor targets never go
through here -- they are found by text, and text does not shift.
"""
from .line_resolver import LAST


class LineNotFound(Exception):
    """Raised when a line number falls outside the current document."""


class PositionTracker:
    """Tracks which original line sits at each current position."""

    def __init__(self, line_count):
        if line_count < 0:
            raise ValueError("line_count must not be negative")
        # Stable ids are the original 1-based line numbers.
        self.ids = list(range(1, line_count + 1))
        self._next_id = line_count + 1

    # ── Queries ──────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.ids)

    def current_index(self, line_id):
        """Current 1-based position of a stable id, or None if deleted."""
        try:
            return self.ids.index(line_id) + 1
        except ValueError:
            return None

    def resolve(self, n):
        """
        Turn a spoken line number into a stable line id.

        n is 1-based, or LAST (-1) for the last line. Returns a dict carrying
        both the position asked for and the line's original number, so the
        demo can say "line 2 (was line 3)".
        """
        if not self.ids:
            raise LineNotFound("the document has no lines left")

        index = len(self.ids) - 1 if n == LAST else n - 1
        if index < 0 or index >= len(self.ids):
            raise LineNotFound(
                f"line {n} does not exist — the document has "
                f"{len(self.ids)} line{'s' if len(self.ids) != 1 else ''}"
            )

        line_id = self.ids[index]
        return {
            "line_id": line_id,
            "current": index + 1,
            "original": line_id,
            "moved": line_id != index + 1,
        }

    # ── Edits ────────────────────────────────────────────────────────────────
    def delete(self, line_id):
        """Remove a line; every line below it shifts up by one."""
        try:
            self.ids.remove(line_id)
        except ValueError:
            raise LineNotFound(f"line id {line_id} is already gone") from None

    def insert_after(self, line_id):
        """Add a new line below `line_id` and return its stable id."""
        new_id = self._next_id
        self._next_id += 1
        position = self.ids.index(line_id) + 1 if line_id in self.ids else len(self.ids)
        self.ids.insert(position, new_id)
        return new_id

    def apply(self, action, line_id):
        """
        Record the effect of an edit on line numbering.

        Only `delete` and `insert-after` move lines. Formatting actions
        (bold, italic, ...) leave the line list untouched.
        """
        if action == "delete":
            self.delete(line_id)
        elif action == "insert-after":
            return self.insert_after(line_id)
        return None

    def snapshot(self):
        """Current position -> original line number, for the demo UI."""
        return {i + 1: line_id for i, line_id in enumerate(self.ids)}
