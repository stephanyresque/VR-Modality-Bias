"""Caption-text helpers shared across scripts."""

from __future__ import annotations

from collections import Counter

__all__ = ["detect_loop"]


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "by", "for",
    "with", "from", "is", "are", "was", "were", "be", "been", "being", "as",
    "it", "its", "this", "that", "these", "those", "there", "their", "they",
    "he", "she", "him", "her", "his", "hers", "you", "your",
    # punctuation-only artifacts after .strip()
    "", ".", ",",
})


def _norm(tok: str) -> str:
    """Lowercase + strip trailing punctuation. Used by both loop checks."""
    return tok.lower().strip(".,;:!?\"'()[]")


def detect_loop(text: str) -> tuple[bool, str]:
    """Return ``(has_loop, why)`` for a generated caption.

    Two heuristics — either triggers the flag:

      (a) tail-window unigram dominance — some non-stopword unigram
          occupies at least 4 of the last 20 tokens.
      (b) trailing consecutive repeat — some token repeats at least 3
          times in a row anywhere in the last 30 tokens (catches the
          canonical ``"allowed, allowed, allowed"`` / ``"the room. the room."``
          tail-loop).

    Stopwords (``the/a/of/and/...``) are skipped in (a) so normal English
    prose doesn't false-fire. (b) ignores trailing punctuation when
    comparing (``"allowed"`` == ``"allowed,"``). The threshold is
    intentionally permissive — flag suspicion; eyeball the actual caption.

    Empty / very short captions (< 6 tokens) return ``(False, "")``.

    Tests for this function live in ``tests/test_decode_sweep_loop.py``.
    """
    words = text.split()
    if len(words) < 6:
        return False, ""

    # (b) consecutive repeats — the canonical 'X, X, X' tail.
    tail_for_b = words[-30:] if len(words) > 30 else words
    for i in range(len(tail_for_b)):
        if not _norm(tail_for_b[i]):
            continue
        run = 1
        j = i + 1
        while j < len(tail_for_b) and _norm(tail_for_b[j]) == _norm(tail_for_b[i]):
            run += 1
            j += 1
        if run >= 3:
            return True, f"consecutive '{_norm(tail_for_b[i])}' x{run}"

    # (a) tail-window unigram dominance — walk most_common from the top
    # until we find a non-stopword candidate that meets the threshold.
    # The naive top-1 check misses cases like "the room. the room. the room."
    # because "the" and "room" tie at 4 and Counter returns the stopword
    # first by insertion order.
    tail_for_a = [_norm(w) for w in words[-20:]]
    counter = Counter(tail_for_a)
    for tok, count in counter.most_common():
        if count < 4:
            break
        if tok and tok not in _STOPWORDS:
            return True, f"'{tok}' x{count} in last 20 tokens"

    return False, ""
