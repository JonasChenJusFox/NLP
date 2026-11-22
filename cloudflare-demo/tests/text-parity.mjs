import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { createTextProcessor, pythonLower } from '../edge/text.mjs';

const fixture = JSON.parse(fs.readFileSync(new URL('./text-fixtures.json', import.meta.url), 'utf8'));
const processors = Object.fromEntries(Object.entries(fixture.profiles).map(([name, spec]) => [name, createTextProcessor(spec)]));
let comparisons = 0, perturbations = 0;
const same = (actual, expected, name) => { assert.deepStrictEqual(actual, expected, name); comparisons++; };
for (let index = 0; index < fixture.cases.length; index++) {
  const entry = fixture.cases[index], prefix = `case ${index} ${JSON.stringify(entry.input.slice(0, 100))}`;
  same(processors.trained.tokenize(entry.input), entry.tokens, `${prefix}: tokenize`);
  same(processors.trained.tokenize(entry.input, false), entry.plain_tokens, `${prefix}: tokenize(false)`);
  same(processors.trained.normalize(entry.input), entry.normalized, `${prefix}: normalize`);
  for (const edit of entry.edits) {
    const changed = processors.trained.perturb(entry.input, edit.method, edit.seed);
    same(changed, edit.result, `${prefix}: ${edit.method} seed=${edit.seed}`);
    same(processors.trained.normalize(changed.text), edit.normalized, `${prefix}: normalized ${edit.method}`);
    perturbations++;
  }
}
for (const entry of fixture.special) same(processors[entry.profile].normalize(entry.input), entry.result,
  `normalizer ${entry.profile}: ${JSON.stringify(entry.input)}`);
for (let index = 0; index < fixture.lower.length; index++) {
  const entry = fixture.lower[index];
  if (entry.sha256_utf16le) {
    const input = Array.from({ length: entry.count }, (_, i) => String.fromCodePoint(entry.start + i)).join('');
    const digest = createHash('sha256').update(Buffer.from(pythonLower(input), 'utf16le')).digest('hex');
    same(digest, entry.sha256_utf16le, `Python Unicode lowercase ${index}`);
  } else same(pythonLower(entry.input), entry.result, `Python Unicode lowercase ${index}`);
}
console.log(JSON.stringify({ ok: true, cases: fixture.cases.length, perturbations, exact_comparisons: comparisons,
  normalizer_profiles: Object.keys(processors).length, lowercase_cases: fixture.lower.length,
  reference_sha256: fixture.reference_sha256 }, null, 2));
