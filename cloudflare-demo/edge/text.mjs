// A dependency-free port of the frozen backend/text.py. Unicode character
// classes below are generated with the frozen Python runtime (Unicode 13).
import { PYTHON_WORD_CLASS, PYTHON_DIGIT_CLASS, pythonLower } from './unicode.mjs';
export { PYTHON_WORD_CLASS, PYTHON_DIGIT_CLASS, pythonLower } from './unicode.mjs';

export const METHODS = Object.freeze(['substitution', 'punctuation', 'repetition']);
const SUBSTITUTIONS = { a: '4', e: '€', i: '1', o: '0', s: '$' };
const REVERSE = { '4': 'a', '€': 'e', '1': 'i', '0': 'o', '$': 's' };
// Python re.I expands [a-z] to these four extra Unicode case equivalents.
const LETTER_BODY = 'A-Za-z\u0130\u0131\u017f\u212a';
const LETTER = `[${LETTER_BODY}]`;
const ALNUM = `[${LETTER_BODY}0-9€$]`;
const BOUNDARY_ALNUM = `[${LETTER_BODY}0-9]`;
// Python's whitespace also includes U+001C..U+001F, but excludes U+FEFF.
const SPACE_BODY = '\\t-\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000';
const DIGIT = PYTHON_DIGIT_CLASS;
const PROTECTED = [
  `[hH][tT][tT][pP][sSſ]?://[^${SPACE_BODY}]+`,
  `[wW][wW][wW]\\.[^${SPACE_BODY}]+`,
  `[^${SPACE_BODY}<>(),;:@]+@${PYTHON_WORD_CLASS.slice(0, -1)}.\\-]+\\.${LETTER}{2,}`,
  `(?<!${BOUNDARY_ALNUM})\\+?${DIGIT}[${DIGIT.slice(1, -1)} ()-]{5,}${DIGIT}(?!${BOUNDARY_ALNUM})`,
  `(?<!${BOUNDARY_ALNUM})${DIGIT}+(?:[.,:/-]${DIGIT}+)*(?!${BOUNDARY_ALNUM})`,
].join('|');
const WORD = `${ALNUM}(?:\\.${ALNUM}){2,}(?!${ALNUM})|${ALNUM}+(?:['’]${LETTER}+)?`;
const PROTECTED_TOKENS = new RegExp(`(?<protected>${PROTECTED})|(?<word>${WORD})|(?<other>.)`, 'gsu');
const PLAIN_TOKENS = new RegExp(`(?<word>${WORD})|(?<other>.)`, 'gsu');
const HAS_LETTER = new RegExp(LETTER, 'u');
const HAS_UPPER = /[A-ZİK]/u;
const HAS_DIGIT = new RegExp(DIGIT, 'gu');
const ORDINAL = new RegExp(`^${DIGIT}+(?:[sSſ][tT]|[nN][dD]|[rR][dD]|[tT][hH])$`, 'u');

function tokenize(text, protectContent = true) {
  const pattern = protectContent ? PROTECTED_TOKENS : PLAIN_TOKENS;
  pattern.lastIndex = 0;
  const tokens = [];
  let start = 0;
  for (const match of text.matchAll(pattern)) {
    const value = match[0];
    tokens.push({ value, start, protected: match.groups.protected !== undefined,
      word: match.groups.word !== undefined && HAS_LETTER.test(value) });
    start += Array.from(value).length;
  }
  return tokens;
}

// CPython random.Random(integer_seed): init_by_array, MT19937 and
// _randbelow_with_getrandbits. Random.choice consumes rejection draws too.
class PythonRandom {
  constructor(seed) {
    if (typeof seed === 'number' && !Number.isSafeInteger(seed)) {
      throw new RangeError('Seed must be an exact integer.');
    }
    let value = BigInt(seed);
    if (value < 0n) value = -value;
    const key = [];
    do { key.push(Number(value & 0xffffffffn)); value >>= 32n; } while (value);
    this.mt = new Uint32Array(624);
    this.mt[0] = 19650218;
    for (let i = 1; i < 624; i++) {
      const previous = this.mt[i - 1];
      this.mt[i] = (Math.imul(previous ^ (previous >>> 30), 1812433253) + i) >>> 0;
    }
    let i = 1, j = 0;
    for (let count = Math.max(624, key.length); count > 0; count--) {
      const previous = this.mt[i - 1];
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous ^ (previous >>> 30), 1664525)) + key[j] + j) >>> 0;
      if (++i >= 624) { this.mt[0] = this.mt[623]; i = 1; }
      if (++j >= key.length) j = 0;
    }
    for (let count = 623; count > 0; count--) {
      const previous = this.mt[i - 1];
      this.mt[i] = ((this.mt[i] ^ Math.imul(previous ^ (previous >>> 30), 1566083941)) - i) >>> 0;
      if (++i >= 624) { this.mt[0] = this.mt[623]; i = 1; }
    }
    this.mt[0] = 0x80000000;
    this.index = 624;
  }

  nextUint32() {
    if (this.index >= 624) {
      for (let i = 0; i < 624; i++) {
        const joined = (this.mt[i] & 0x80000000) | (this.mt[(i + 1) % 624] & 0x7fffffff);
        this.mt[i] = this.mt[(i + 397) % 624] ^ (joined >>> 1) ^ ((joined & 1) ? 0x9908b0df : 0);
      }
      this.index = 0;
    }
    let value = this.mt[this.index++];
    value ^= value >>> 11;
    value ^= (value << 7) & 0x9d2c5680;
    value ^= (value << 15) & 0xefc60000;
    value ^= value >>> 18;
    return value >>> 0;
  }

  randbelow(length) {
    const bits = length.toString(2).length;
    let candidate;
    do { candidate = this.nextUint32() >>> (32 - bits); } while (candidate >= length);
    return candidate;
  }
}

function perturb(text, method, seed = 42) {
  if (!METHODS.includes(method)) throw new Error(`Unknown change type: ${method}`);
  const tokens = tokenize(text);
  const eligible = [];
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (!token.protected && /^[A-Za-z]{3,}$/.test(token.value)
      && (method !== 'substitution' || /[aeios]/i.test(token.value))) eligible.push(i);
  }
  const selected = eligible.length ? eligible[new PythonRandom(seed).randbelow(eligible.length)] : null;
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    let output = token.value;
    if (index === selected) {
      if (method === 'substitution') {
        const letter = Array.from('eaios').find(c => pythonLower(output).includes(c));
        output = output.replace(new RegExp(letter, 'i'), match => SUBSTITUTIONS[pythonLower(match)]);
      } else if (method === 'punctuation') output = Array.from(output).join('.');
      else output += output.at(-1);
    }
    token.output = output;
    token.changed = output !== token.value;
  }
  return { text: tokens.map(token => token.output).join(''), tokens,
    changed: selected !== null, eligible_word_count: eligible.length,
    changed_word_count: Number(selected !== null) };
}

function codepointCompare(left, right) {
  const a = Array.from(left, c => c.codePointAt(0));
  const b = Array.from(right, c => c.codePointAt(0));
  for (let i = 0; i < Math.min(a.length, b.length); i++) if (a[i] !== b[i]) return a[i] - b[i];
  return a.length - b.length;
}

export function createTextProcessor(spec) {
  const vocabulary = new Map(Object.entries(spec.vocabulary));
  const minDf = spec.min_df ?? 2;
  const protectContent = spec.protect_content ?? true;
  function protectedWord(value) {
    return protectContent && (HAS_UPPER.test(value) || value.includes("'") || value.includes('’')
      || ORDINAL.test(value) || Array.from(value.matchAll(HAS_DIGIT)).length > 1 || /[2356789]/.test(value));
  }

  function normalize(text) {
    const tokens = tokenize(text, protectContent), records = [];
    for (const token of tokens) {
      const raw = token.value;
      const lower = pythonLower(raw);
      let output = raw, candidates = [], reason;
      if (token.protected || (token.word && protectedWord(raw))) reason = 'protected';
      else if (!token.word) reason = 'separator';
      else if (vocabulary.has(lower)) reason = 'known_word';
      else {
        const mapped = Array.from(pythonLower(raw.normalize('NFKC')), c => REVERSE[c] ?? c).join('').replaceAll('.', '');
        const chars = Array.from(mapped);
        const possible = new Set([mapped]);
        for (let i = 1; i < chars.length; i++) {
          if (chars[i] === chars[i - 1]) possible.add([...chars.slice(0, i), ...chars.slice(i + 1)].join(''));
        }
        candidates = Array.from(possible).filter(word => word !== lower && (vocabulary.get(word) ?? 0) >= minDf).sort(codepointCompare);
        if (candidates.length === 1) { output = candidates[0]; reason = 'vocabulary_match'; }
        else reason = candidates.length ? 'ambiguous' : 'no_match';
      }
      token.output = output;
      token.changed = output !== raw;
      if (reason !== 'separator' && reason !== 'known_word') records.push({ original: raw, output, reason, candidates, start: token.start });
    }
    return { text: tokens.map(token => token.output).join(''), tokens, records };
  }

  return { tokenize, perturb, normalize };
}
