"""Record existing Python API responses for hosted-adapter regression checks."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from backend.api import create_app

texts = [
    'Congratulations! Claim your free prize now. Call 07700900123 to collect your reward.',
    'Are you free for coffee tomorrow? Please call me after work.',
    'Your free trial is ready. Visit https://example.com/free or call +1 212-555-0199. Email hello@example.com.',
    'claim fr€e f.r.e.e freee prizee 1st WIN today',
    '😀 hello free café İıſK Σ ΟΣ ΟΣΑ 👨‍👩‍👧‍👦',
    '123 https://example.com/ hello@example.com',
    '   free\t\tprize\x1cnow\u0085 claim 𝟡𝟡𝟡',
    '__proto__ constructor toString free',
    'a',
    '中文短信 free tomorrow'
]
fixtures = []
with TestClient(create_app()) as client:
    for text in texts:
        for method in ['substitution', 'punctuation', 'repetition']:
            for seed in [42, 4294967295]:
                request = dict(text=text, method=method, seed=seed)
                result = client.post('/api/analyze', json=request)
                result.raise_for_status()
                fixtures.append(dict(request=request, expected=result.json()))
path = Path(__file__).resolve().parents[1] / 'tests/api-fixtures.json'
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(fixtures, ensure_ascii=False, separators=(',', ':')) + '\n')
print(f'Wrote {len(fixtures)} synthetic API reference responses')
