import assert from 'node:assert/strict';
import {readFile, writeFile} from 'node:fs/promises';

const url = new URL(process.argv[2]);
assert.equal(url.protocol, 'https:');
const routes = [];
for (const route of ['/', '/app.js', '/style.css', '/api/health']) {
  // Deliberately send no authentication header or cookie.
  const response = await fetch(new URL(route, url), {redirect: 'error'});
  assert.equal(response.status, 200, route);
  const body = await response.text();
  if (route === '/') assert.match(body, /SMS Spam Robustness/);
  if (route === '/api/health') assert.deepEqual(JSON.parse(body), {
    status: 'ok', models_ready: true, model_id: '9439f39e3b9e7f45'
  });
  routes.push({route, status: response.status});
}
const fixtures = JSON.parse(await readFile(new URL('./api-fixtures.json', import.meta.url), 'utf8'));
let maxError = 0;
for (const index of [0, 2, 4, 12, 24, 42, 48, 54]) {
  const fixture = fixtures[index];
  const response = await fetch(new URL('/api/analyze', url), {
    method: 'POST', redirect: 'error',
    headers: {'content-type': 'application/json'}, body: JSON.stringify(fixture.request)
  });
  assert.equal(response.status, 200, `fixture ${index}`);
  const actual = await response.json();
  for (const system of ['majority', 'A', 'B', 'C']) for (const condition of ['original', 'modified']) {
    const expected = fixture.expected.predictions[system][condition].score;
    const error = Math.abs(actual.predictions[system][condition].score - expected);
    maxError = Math.max(maxError, error);
    assert.ok(error <= 1e-10, `fixture ${index} ${system} ${condition}`);
    actual.predictions[system][condition].score = expected;
  }
  assert.deepEqual(actual, fixture.expected, `fixture ${index}`);
}
const boundaryInputs = [
  'free prize now '.repeat(143).slice(0, 2000),
  'a'.repeat(2000),
  '😀'.repeat(2000)
];
for (const text of boundaryInputs) {
  const response = await fetch(new URL('/api/analyze', url), {
    method: 'POST', redirect: 'error', headers: {'content-type': 'application/json'},
    body: JSON.stringify({text, method: 'punctuation', seed: 42})
  });
  assert.equal(response.status, 200, `2000-character boundary input`);
  const actual = await response.json();
  assert.equal(actual.original_text, text);
  assert.equal(actual.model_id, '9439f39e3b9e7f45');
}
const report = {
  passed: true, url: url.origin, platform: 'Cloudflare Workers (direct deployment)',
  authentication_sent: false, model_id: '9439f39e3b9e7f45', routes,
  api_reference_cases: 8, boundary_cases: boundaryInputs.length,
  max_score_error: maxError, browser_visual_check: false
};
if (process.argv[3]) await writeFile(process.argv[3], JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
