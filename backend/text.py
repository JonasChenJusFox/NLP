"""Shared transformation rules. Nothing is learnt from dev, test or demo input."""

import hashlib
import random
import re
import unicodedata
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


METHODS = ("substitution", "punctuation", "repetition")
SUBSTITUTIONS = {"a": "4", "e": "€", "i": "1", "o": "0", "s": "$"}
REVERSE = str.maketrans({value: key for key, value in SUBSTITUTIONS.items()})
PROTECTED_PATTERN = (
    r"https?://[^\s]+|www\.[^\s]+|[^\s<>(),;:@]+@[\w.-]+\.[a-z]{2,}"
    r"|(?<![a-z0-9])\+?\d[\d ()-]{5,}\d(?![a-z0-9])"
    r"|(?<![a-z0-9])\d+(?:[.,:/-]\d+)*(?![a-z0-9])"
)
# Only chains of single letters are joined across dots, e.g. f.r.e.e.
# Ordinary word boundaries, e.g. 'free.prize', are never joined.
WORD_PATTERN = r"[a-z0-9€$](?:\.[a-z0-9€$]){2,}(?![a-z0-9€$])|[a-z0-9€$]+(?:['’][a-z]+)?"
TOKEN_PATTERN = re.compile(
    rf"(?P<protected>{PROTECTED_PATTERN})|(?P<word>{WORD_PATTERN})|(?P<other>.)",
    re.IGNORECASE | re.DOTALL,
)


def tokenize(text, protect_content=True):
    pattern = TOKEN_PATTERN if protect_content else re.compile(
        rf"(?P<word>{WORD_PATTERN})|(?P<other>.)", re.IGNORECASE | re.DOTALL
    )
    return [
        {"value": m.group(), "start": m.start(), "protected": m.lastgroup == "protected",
         "word": m.lastgroup == "word" and bool(re.search("[a-z]", m.group(), re.I))}
        for m in pattern.finditer(text)
    ]


def message_seed(seed, message_id):
    """Stable per-message randomness, independent of row ordering and labels."""
    digest = hashlib.sha256(f"{seed}:{message_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def perturb(text, method, seed=42):
    """Apply exactly one edit to at most one eligible word; protect entities."""
    if method not in METHODS:
        raise ValueError(f"Unknown change type: {method}")
    tokens = tokenize(text)
    eligible = [i for i, token in enumerate(tokens)
                if not token["protected"] and token["value"].isascii()
                and re.fullmatch(r"[a-z]{3,}", token["value"], re.I)
                and (method != "substitution" or re.search("[aeios]", token["value"], re.I))]
    selected = random.Random(seed).choice(eligible) if eligible else None
    for index, token in enumerate(tokens):
        output = token["value"]
        if index == selected:
            if method == "substitution":
                # Prefer e so the documented free -> fr€e example is exact.
                letter = next(char for char in "eaios" if char in output.lower())
                output = re.sub(letter, lambda m: SUBSTITUTIONS[m.group().lower()],
                                output, count=1, flags=re.I)
            elif method == "punctuation":
                output = ".".join(output)
            else:
                output += output[-1]
        token.update(output=output, changed=output != token["value"])
    return {"text": "".join(t["output"] for t in tokens), "tokens": tokens,
            "changed": selected is not None, "eligible_word_count": len(eligible),
            "changed_word_count": int(selected is not None)}


class ConservativeNormalizer(BaseEstimator, TransformerMixin):
    """Bounded rule candidates, training document-frequency evidence, no spellchecker."""

    def __init__(self, min_df=2, protect_content=True):
        self.min_df = min_df
        self.protect_content = protect_content

    def fit(self, X, y=None):
        self.vocabulary_ = Counter()
        for text in X:
            self.vocabulary_.update({
                token["value"].lower() for token in tokenize(text)
                if token["word"] and re.fullmatch("[a-z]{2,}", token["value"], re.I)
            })
        return self

    def transform(self, X):
        check_is_fitted(self, "vocabulary_")
        return [self.normalize(text)["text"] for text in X]

    def _protected_word(self, token):
        value = token["value"]
        # Deliberately conservative: all title-case, uppercase and mixed-case
        # tokens are possible names/abbreviations, including sentence starts.
        return self.protect_content and (
            any(c.isupper() for c in value) or "'" in value or "’" in value
            or len(re.findall(r"\d", value)) > 1 or bool(re.search(r"[2356789]", value))
        )

    def normalize(self, text):
        check_is_fitted(self, "vocabulary_")
        tokens = tokenize(text, self.protect_content)
        records = []
        for token in tokens:
            raw = token["value"]
            output = raw
            candidates = []
            if token["protected"] or (token["word"] and self._protected_word(token)):
                reason = "protected"
            elif not token["word"]:
                reason = "separator"
            elif raw.lower() in self.vocabulary_:
                reason = "known_word"
            else:
                mapped = unicodedata.normalize("NFKC", raw).lower().translate(REVERSE)
                mapped = mapped.replace(".", "")
                # Delete one repeated character, not every repeated run. Thus
                # freee -> free is possible without corrupting the 'ee' in free.
                possible = {mapped}
                possible.update(mapped[:i] + mapped[i + 1:]
                                for i in range(1, len(mapped)) if mapped[i] == mapped[i - 1])
                candidates = sorted(word for word in possible if word != raw.lower()
                                    and self.vocabulary_[word] >= self.min_df)
                if len(candidates) == 1:
                    output, reason = candidates[0], "vocabulary_match"
                else:
                    reason = "ambiguous" if candidates else "no_match"
            token.update(output=output, changed=output != raw)
            if reason not in {"separator", "known_word"}:
                records.append({"original": raw, "output": output, "reason": reason,
                                "candidates": candidates, "start": token["start"]})
        return {"text": "".join(t["output"] for t in tokens), "tokens": tokens, "records": records}
