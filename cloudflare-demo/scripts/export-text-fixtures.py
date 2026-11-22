"""Reference every text field against the original Python implementation."""
import csv
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.models import load_bundle
from backend.text import ConservativeNormalizer, METHODS, perturb, tokenize


def spec(normalizer):
    return {"vocabulary": dict(normalizer.vocabulary_), "min_df": normalizer.min_df,
            "protect_content": normalizer.protect_content}


def main():
    bundle = load_bundle(ROOT / "artifacts/models.joblib")
    normalizers = {
        "trained": bundle["models"]["C"].named_steps["normalize"],
        "toy": ConservativeNormalizer().fit(["free offer prize hello coffee ist ath info ssl kil" ] * 2),
        "single": ConservativeNormalizer().fit(["free"]),
        "ambiguous": ConservativeNormalizer().fit(["free freee"] * 2),
        "unprotected": ConservativeNormalizer(protect_content=False).fit([
            "free offer prize hello coffee ist ath info ssl kil İİ ſſ KK ıı"
        ] * 2),
    }
    texts = [
        "free", "0ffer", "pr1ze", "coffeee", "hello", "fr€ee", "freee", "freeee",
        "claim free prize and coffee today", "J0nas FR€E Fr€e iPhone F.R.E.E",
        "https://example.com/fr€e?a=1 hi+fr€e@example.org +1 212-555-0199 12345 12.50",
        "1st 2nd 3rd 4th 11th 21st 101st", "1nfo fr€e", "unseenwordd",
        "free.prize hello,coffee\nfree!", "12345 😀", "ſſſ İİİ KKK ııı",
        "😀 🧑🏾‍💻 fr€e c.o.f.f.e.e 0ffer!", "f.r.e.e.prize f.r.e.ee f.r.e.e x.y.z.",
        "f.r.e.e! F.R.E.E! fr€e? pr1ze; a4a oooffer coffffee", "",
        "www.example.com/fr€e HTTPS://EXAMPLE.COM\nhttpſ://example.com/fr€e",
        "hello@example.ſſ fr€e@例.测试 hi@ａａ.KK hi@𞓐.ab hello@aa.İİ",
        "hi@a\U0001e4d0.ab hi@a\U0001e290.ab hi@a\U0001e900.ab",
        "fr€e\x1cwww.example.com\x1dfr€e\ufeffwww.example.com\x85hello\u00a0bye",
        "١٢٣ ۱۲۳ １２３ +١ ٢١٢-٥٥٥-٠١٩٩ 12.50 3/14 4:30 01-02-2026",
        "\U00011f50\U00011f51\U00011f52 \U0001e4f0\U0001e4f1\U0001e4f2",
        "1ſt ١st ١١th 1ST 4TH 1000€ fr€e1 pr1ze h3llo abc7",
        "İİ ſſ KK ıı iİi ııı i̇i̇ can't won’t fr€e’s f.r.e.e's",
        "a 0 $ € 12 abc abcd abcde abcdef abcdefg abcdefgh abcdefghi",
        "abcabc eee aei ios 0123 1a a1 ab1 1ab $ale | $$$ | 000",
        "\r\nfree\tprize\vhello\fcoffee\u2028offer\u2029info",
        "hi@example.com +44 (020) 1234-5678 hi+tag@example.org fr€e",
        "a12345b a+1 234 567 890b K12 ſ12 İ12 ı12 12K 12ſ 12İ 12ı",
        "http://example.com/🍕?free=1 www.example.com. free@example.net, 2026-11-22",
        "Aa EA iO sSS claim hello prize free offer coffee", "007s abc-123 foo.1.2.3 x.y.z9 x.y.z0",
    ]
    # Use development inputs for implementation parity, never fit or tune on them.
    with (ROOT / "data/processed/splits.csv").open(newline="", encoding="utf-8") as handle:
        dev = [row["text"] for row in csv.DictReader(handle) if row["split"] == "dev"]
    random.Random(157).shuffle(dev)
    texts.extend(dev[:220])
    # Deterministic mixed protected entities and Unicode fuzz.
    chunks = ["free", "Fr€e", "coffeee", "http://e.co/a", "x@aa.co", "١٢٣", "😀", "İİ", "ſſ",
              "a.b.c", "ab.cd.ef", "1st", "+1 212-555-0199", "1nfo", "\x1c", "\ufeff", "\n", " ", "_", "é"]
    rng = random.Random(759)
    for _ in range(100):
        texts.append(" ".join(rng.choices(chunks, k=rng.randrange(1, 14))))
    seeds = [0, 1, 2, 42, 91, 2147483648, 4294967295, 18446744073709551615]
    cases = []
    for index, text in enumerate(texts):
        edits = []
        for offset, method in enumerate(METHODS):
            seed = seeds[(index + offset) % len(seeds)]
            edited = perturb(text, method, seed)
            edits.append({"method": method, "seed": str(seed), "result": edited,
                          "normalized": normalizers["trained"].normalize(edited["text"])})
        cases.append({"input": text, "tokens": tokenize(text), "plain_tokens": tokenize(text, False),
                      "normalized": normalizers["trained"].normalize(text), "edits": edits})
    special = []
    for name, normalizer in normalizers.items():
        for text in texts[:37]:
            special.append({"profile": name, "input": text, "result": normalizer.normalize(text)})
    # Lowercasing is also used before model TF-IDF analysis. Cover all newly
    # assigned case mappings plus contextual sigma through the full code space.
    lower = []
    for start in range(0, 0x110000, 4096):
        value = "".join(map(chr, range(start, min(start + 4096, 0x110000))))
        # Hash UTF-16 code units so the reference remains compact and includes
        # isolated surrogate codepoints without UTF-8 replacement differences.
        lower.append({"start": start, "count": len(value), "sha256_utf16le": hashlib.sha256(
            value.lower().encode("utf-16-le", errors="surrogatepass")).hexdigest()})
    sigma_chars = [chr(i) for i in range(0x110000)
                   if chr(i).islower() or chr(i).isupper() or chr(i).istitle()
                   or unicodedata.category(chr(i)) in {"Mn", "Me", "Cf", "Lm", "Sk"}
                   or chr(i) in "\u0027\u003a\u00b7"]
    for index in range(0, len(sigma_chars), 100):
        value = "|".join("AΣ" + char + "A " + " " + char + "Σ" for char in sigma_chars[index:index + 100])
        lower.append({"input": value, "result": value.lower()})
    result = {"reference": "backend/text.py", "reference_sha256": hashlib.sha256((ROOT / "backend/text.py").read_bytes()).hexdigest(),
              "profiles": {name: spec(normalizer) for name, normalizer in normalizers.items()},
              "cases": cases, "special": special, "lower": lower}
    target = ROOT / "cloudflare-demo/tests/text-fixtures.json"
    target.write_text(json.dumps(result, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"target": str(target), "cases": len(cases), "edits": sum(len(case["edits"]) for case in cases),
                      "normalizer_profiles": len(normalizers), "unicode_lower_cases": len(lower), "bytes": target.stat().st_size}))


if __name__ == "__main__":
    main()
