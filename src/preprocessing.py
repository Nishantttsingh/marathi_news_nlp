"""Single source of truth for Marathi text preprocessing.

The Streamlit "NLP Preprocessing" tab and the TF-IDF vectorizer both call the
functions below, so the text shown in the UI is exactly what the model sees.

Implemented: Unicode NFC, zero-width character removal, Devanagari->ASCII digit
mapping, Latin lower-casing, punctuation removal, whitespace collapsing,
regex tokenization, stopword removal.
NOT implemented: stemming / lemmatization (no reliable Marathi stemmer is
available offline; a rule-based one would be unvalidated).
"""
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

STOPWORD_FILE = Path(__file__).resolve().parent.parent / "data" / "marathi_stopwords.txt"

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_DIGITS = str.maketrans("\u0966\u0967\u0968\u0969\u096a\u096b\u096c\u096d\u096e\u096f", "0123456789")
# Token = run of Devanagari letters/vowel signs/virama (U+0900-0963, U+0971-097F), Latin letters, ASCII digits.
# (Python's \w would split words at Devanagari vowel signs, so an explicit class is used.)
_TOKEN = re.compile(r"[\u0900-\u0963\u0971-\u097FA-Za-z0-9]+")
_NON_TOKEN = re.compile(r"[^\u0900-\u0963\u0971-\u097FA-Za-z0-9\s]")
_SPACES = re.compile(r"\s+")

# (description, function) - applied in order; the UI reports only steps that changed the text.
_STEPS = [
    ("Unicode NFC normalization (canonical composition of Devanagari code points)", lambda s: unicodedata.normalize("NFC", s)),
    ("Removed zero-width characters (ZWJ / ZWNJ / ZWSP / BOM)", lambda s: _ZERO_WIDTH.sub("", s)),
    ("Mapped Devanagari digits (०-९) to ASCII digits (0-9)", lambda s: s.translate(_DIGITS)),
    ("Lower-cased Latin letters", lambda s: s.lower()),
    ("Replaced punctuation/symbols (including danda '।') with spaces", lambda s: _NON_TOKEN.sub(" ", s)),
    ("Collapsed repeated whitespace and trimmed ends", lambda s: _SPACES.sub(" ", s).strip()),
]


@lru_cache(maxsize=1)
def load_stopwords(path: str = str(STOPWORD_FILE)) -> frozenset:
    words = set()
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(unicodedata.normalize("NFC", line))
    return frozenset(words)


def normalize(text) -> str:
    s = "" if text is None else str(text)
    for _, fn in _STEPS:
        s = fn(s)
    return s


def tokenize(normalized: str) -> list:
    return _TOKEN.findall(normalized)


def remove_stopwords(tokens: list):
    sw = load_stopwords()
    return [t for t in tokens if t not in sw], [t for t in tokens if t in sw]


def tokenize_and_filter(normalized: str) -> list:
    """Callable handed to TfidfVectorizer(tokenizer=...). Input is already normalized."""
    return remove_stopwords(tokenize(normalized))[0]


def process(text) -> dict:
    """Full step-by-step trace used by the UI."""
    original = "" if text is None else str(text)
    s, applied = original, []
    for desc, fn in _STEPS:
        new = fn(s)
        if new != s:
            applied.append(desc)
        s = new
    tokens = tokenize(s)
    kept, removed = remove_stopwords(tokens)
    return {"original": original, "normalized": s, "applied_steps": applied, "tokens": tokens,
            "removed_stopwords": removed, "tokens_after": kept, "final_text": " ".join(kept)}
