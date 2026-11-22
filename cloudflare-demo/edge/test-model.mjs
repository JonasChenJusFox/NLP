import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createPredictor, wordNgrams, characterNgrams } from './model.mjs';

const folder = new URL('.', import.meta.url);
const data = JSON.parse(await fs.readFile(new URL('model-data.json', folder), 'utf8'));
const fixtures = JSON.parse(await fs.readFile(new URL('model-fixtures.json', folder), 'utf8'));
const model = createPredictor(data);
assert.equal(model.modelId, fixtures.model_id);
assert.deepEqual(wordNgrams('Hello free prize'), ['hello', 'free', 'prize', 'hello free', 'free prize']);
// These became letters/digits after the frozen Python Unicode 13 database.
assert.deepEqual(wordNgrams('free\u1c89prize'), ['free', 'prize', 'free prize']);
assert.deepEqual(wordNgrams('free\u{10d40}prize'), ['free', 'prize', 'free prize']);
assert.deepEqual(characterNgrams('😀😀😀', 3, 3), ['😀😀😀']);
assert.deepEqual(characterNgrams('a\x1c\x1fb', 3, 3), ['a b']);
assert.equal(model.score('majority', 'claim free prize'), 0);
let maxError = 0;
for (const fixture of fixtures.rows) {
  const result = model.predict(fixture.system, fixture.model_input);
  const error = Math.abs(result.score - fixture.score);
  maxError = Math.max(maxError, error);
  assert.ok(error <= 1e-10, `${fixture.case_id}/${fixture.system} score mismatch ${error} for ${JSON.stringify(fixture.model_input)}`);
  assert.equal(result.label, fixture.prediction, `${fixture.case_id}/${fixture.system} label mismatch`);
}
console.log(JSON.stringify({ model_id: model.modelId, fixture_rows: fixtures.rows.length, max_absolute_error: maxError, labels_equal: true }));
