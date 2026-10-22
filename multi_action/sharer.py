"""
Target sharer.

A chunk may name an action but no target:

    "first line bold-um italic-um pannu"
        chunk 1 -> bold   + line 1
        chunk 2 -> italic + (nothing)

The second chunk borrows the first chunk's target. Sharing only ever reaches
backwards, and only fills a target that is missing -- a chunk that found its
own target keeps it.
"""
import copy


def share_targets(actions):
    """
    Fill missing targets from the nearest earlier chunk that has one.

    `actions` is a list of dicts with at least an "action" and a "target" key.
    Returns a new list; the input is not modified. Each filled entry is marked
    with shared_target=True so the demo can show where the target came from.
    """
    filled = []
    last_target = None
    last_source = None

    for index, item in enumerate(actions):
        entry = copy.deepcopy(item)

        if entry.get("target"):
            last_target = copy.deepcopy(entry["target"])
            last_source = index
            entry["shared_target"] = False
        elif entry.get("action") and last_target is not None:
            entry["target"] = copy.deepcopy(last_target)
            entry["shared_target"] = True
            entry["shared_from_chunk"] = last_source
        else:
            entry["shared_target"] = False

        filled.append(entry)

    return filled
