/* Exact exported TF-IDF/LR inference. No fitting, network or runtime packages. */

import { PYTHON_WORD_CLASS, pythonLower } from './text.mjs';

const PYTHON_WHITESPACE_RUN = /[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]{2,}/gu;
// Python (?u) \w is alphanumeric plus underscore, excluding combining marks.
const WORD_TOKENS = new RegExp(`${PYTHON_WORD_CLASS}{2,}`, 'gu');

function floatsFromBase64(encoded, count) {
  const binary = atob(encoded);
  if (binary.length !== count * 8) throw new Error('Invalid Float64 model array length.');
  const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const result = new Float64Array(count);
  for (let index = 0; index < count; index++) result[index] = view.getFloat64(index * 8, true);
  return result;
}

function sigmoid(value) {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exp = Math.exp(value);
  return exp / (1 + exp);
}

export function wordNgrams(text, minimum = 1, maximum = 2) {
  const tokens = Array.from(pythonLower(text).matchAll(WORD_TOKENS), match => match[0]);
  const result = [];
  for (let length = minimum; length <= Math.min(maximum, tokens.length); length++) {
    for (let start = 0; start <= tokens.length - length; start++) {
      result.push(tokens.slice(start, start + length).join(' '));
    }
  }
  return result;
}

export function characterNgrams(text, minimum = 3, maximum = 5) {
  const characters = Array.from(pythonLower(text).replace(PYTHON_WHITESPACE_RUN, ' '));
  const result = [];
  for (let length = minimum; length <= Math.min(maximum, characters.length); length++) {
    for (let start = 0; start <= characters.length - length; start++) {
      result.push(characters.slice(start, start + length).join(''));
    }
  }
  return result;
}

export function createPredictor(data) {
  if (data.format_version !== 1) throw new Error('Unsupported exported model format.');
  const models = new Map();
  for (const system of ['A', 'B', 'C']) {
    const source = data.models[system];
    if (!source || source.features.length !== source.feature_count) throw new Error('Incomplete model vocabulary.');
    const vocabulary = new Map(source.features.map((feature, index) => [feature, index]));
    if (vocabulary.size !== source.feature_count) throw new Error('Duplicate exported model features.');
    models.set(system, {
      vocabulary,
      idf: floatsFromBase64(source.idf_b64, source.feature_count),
      coefficients: floatsFromBase64(source.coef_b64, source.feature_count),
      intercept: source.intercept,
      analyzer: source.analyzer,
      ngramRange: source.ngram_range
    });
  }

  function threshold(system) {
    const selected = data.selected[system];
    if (!selected) throw new Error('Unknown model system.');
    return selected.threshold;
  }

  function score(system, text) {
    if (typeof text !== 'string') throw new TypeError('Model text must be a string.');
    if (system === 'majority') return data.majority.spam_score;
    const model = models.get(system);
    if (!model) throw new Error('Unknown model system.');
    const ngrams = model.analyzer === 'char'
      ? characterNgrams(text, ...model.ngramRange)
      : wordNgrams(text, ...model.ngramRange);
    const frequencies = new Map();
    for (const feature of ngrams) {
      const index = model.vocabulary.get(feature);
      if (index !== undefined) frequencies.set(index, (frequencies.get(index) || 0) + 1);
    }
    let squaredNorm = 0;
    let weighted = 0;
    // sklearn CSR data are sorted by feature index. Matching the order also
    // keeps floating-point summation as close as possible to native inference.
    const indices = Array.from(frequencies.keys()).sort((a, b) => a - b);
    for (const index of indices) {
      const value = (1 + Math.log(frequencies.get(index))) * model.idf[index];
      squaredNorm += value * value;
      weighted += value * model.coefficients[index];
    }
    const normalized = squaredNorm > 0 ? weighted / Math.sqrt(squaredNorm) : 0;
    return sigmoid(model.intercept + normalized);
  }

  function predict(system, text) {
    const probability = score(system, text);
    return { label: probability >= threshold(system) ? 'spam' : 'ham', score: probability };
  }

  return { score, threshold, predict, modelId: data.model_id };
}
