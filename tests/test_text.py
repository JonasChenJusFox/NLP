import pytest

from backend.text import ConservativeNormalizer, METHODS, message_seed, perturb, tokenize


@pytest.fixture
def normalizer():
    return ConservativeNormalizer().fit(["free offer prize hello coffee", "free offer prize hello coffee"])


@pytest.mark.parametrize("method,expected", [
    ("substitution", "fr€e"), ("punctuation", "f.r.e.e"), ("repetition", "freee")
])
def test_documented_edits_and_recovery(normalizer, method, expected):
    result = perturb("free", method)
    assert result["text"] == expected
    assert result["changed_word_count"] == 1
    assert normalizer.transform([expected]) == ["free"]


@pytest.mark.parametrize("text,expected", [("0ffer", "offer"), ("pr1ze", "prize"),
                                          ("coffeee", "coffee"), ("hello", "hello")])
def test_candidates_keep_legitimate_repeated_letters(normalizer, text, expected):
    assert normalizer.transform([text]) == [expected]


@pytest.mark.parametrize("method", METHODS)
def test_single_word_budget_and_determinism(method):
    text = "claim free prize and coffee today"
    first = perturb(text, method, seed=91)
    assert first == perturb(text, method, seed=91)
    assert sum(token["changed"] for token in first["tokens"]) == 1
    assert first["eligible_word_count"] >= 4
    assert message_seed(42, "one") == message_seed(42, "one")
    assert message_seed(42, "one") != message_seed(42, "two")


@pytest.mark.parametrize("method", METHODS)
def test_protected_spans_survive_both_steps(normalizer, method):
    text = "https://example.com/fr€e?a=1 hi+fr€e@example.org +1 212-555-0199 12345 12.50"
    assert perturb(text, method)["text"] == text
    assert normalizer.normalize(text)["text"] == text
    assert "".join(t["value"] for t in tokenize(text)) == text


def test_conservative_names_abbreviations_and_ambiguity(normalizer):
    text = "J0nas FR€E Fr€e iPhone F.R.E.E"
    assert normalizer.transform([text]) == [text]
    ambiguous = ConservativeNormalizer().fit(["free freee", "free freee"])
    result = ambiguous.normalize("fr€ee")
    assert result["text"] == "fr€ee"
    assert result["records"][0]["reason"] == "ambiguous"


def test_vocabulary_does_not_learn_from_prediction(normalizer):
    vocabulary = normalizer.vocabulary_.copy()
    assert normalizer.transform(["unseenwordd"]) == ["unseenwordd"]
    assert normalizer.vocabulary_ == vocabulary
    assert ConservativeNormalizer().fit(["free"]).transform(["fr€e"]) == ["fr€e"]


def test_ordinary_punctuation_does_not_join_words(normalizer):
    text = "free.prize hello,coffee\nfree!"
    assert normalizer.transform([text]) == [text]
    assert not perturb("12345 😀", "substitution")["changed"]


@pytest.mark.parametrize("method", METHODS)
def test_non_ascii_case_equivalents_are_not_eligible(method):
    # Python's IGNORECASE also matches long-s, dotted-i and Kelvin-sign against
    # [a-z]. They must not reach the small ASCII substitution map.
    text = "ſſſ İİİ KKK ııı"
    assert perturb(text, method)["text"] == text
