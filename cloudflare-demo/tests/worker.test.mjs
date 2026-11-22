import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import worker from '../dist/server/index.js';

const origin = 'https://demo.example';
const analyze = payload => worker.fetch(new Request(origin + '/api/analyze', {
  method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify(payload)
}));

test('built artifact serves the complete demo, assets, health and correct route status', async () => {
  const home = await worker.fetch(new Request(origin));
  assert.equal(home.status, 200);
  assert.match(home.headers.get('content-type'), /text\/html/);
  assert.match(await home.text(), /SMS Spam Robustness/);
  for (const file of ['app.js', 'style.css']) {
    const result = await worker.fetch(new Request(`${origin}/${file}`));
    assert.equal(result.status, 200);
    assert.ok((await result.text()).length > 500);
  }
  const head = await worker.fetch(new Request(origin, {method:'HEAD'}));
  assert.equal(head.status, 200);
  assert.equal(await head.text(), '');
  assert.equal((await worker.fetch(new Request(origin + '/unknown'))).status, 404);
  assert.equal((await worker.fetch(new Request(origin + '/api/unknown'))).status, 404);
  assert.equal((await worker.fetch(new Request(origin + '/api/analyze'))).status, 405);
  const health = await worker.fetch(new Request(origin + '/api/health'));
  assert.deepEqual(await health.json(), {status:'ok',models_ready:true,model_id:'9439f39e3b9e7f45'});
});

test('built artifact reproduces Python API tokens, logs, labels and scores', async () => {
  const fixtures = JSON.parse(await readFile(new URL('./api-fixtures.json', import.meta.url), 'utf8'));
  let maxError = 0;
  for (const [index, fixture] of fixtures.entries()) {
    const result = await analyze(fixture.request);
    assert.equal(result.status, 200, `fixture ${index}`);
    const actual = await result.json();
    for (const system of ['majority','A','B','C']) {
      for (const condition of ['original','modified']) {
        const expected = fixture.expected.predictions[system][condition].score;
        const error = Math.abs(actual.predictions[system][condition].score - expected);
        maxError = Math.max(maxError, error);
        assert.ok(error <= 1e-10, `fixture ${index} ${system} ${condition}: ${error}`);
        actual.predictions[system][condition].score = expected;
      }
    }
    assert.deepEqual(actual, fixture.expected, `fixture ${index}`);
  }
  console.log(JSON.stringify({api_reference_cases:fixtures.length,max_score_error:maxError}));
});

test('API rejects malformed and out-of-budget requests and honors omitted defaults', async () => {
  const invalid = [null, [], {}, {text:''}, {text:' \x1c\u0085'}, {text:123}, {text:'a'.repeat(2001)},
    {text:'hello',method:'unknown'}, {text:'hello',method:null}, {text:'hello',seed:-1},
    {text:'hello',seed:4294967296}, {text:'hello',seed:0.5}, {text:'hello',seed:null},
    {text:'hello',extra:'unexpected'}];
  for (const payload of invalid) assert.equal((await analyze(payload)).status, 422, JSON.stringify(payload));
  const malformed = await worker.fetch(new Request(origin + '/api/analyze', {method:'POST',body:'{'}));
  assert.equal(malformed.status, 400);
  const large = await worker.fetch(new Request(origin + '/api/analyze', {method:'POST',body:'x'.repeat(32769)}));
  assert.equal(large.status, 413);
  const result = await analyze({text:'hello free prize'});
  assert.equal(result.status, 200);
  assert.equal((await result.json()).seed, 42);
});
