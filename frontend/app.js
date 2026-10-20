/* Browser-only prototype. Replace the demo vocabulary with training vocabulary
   when integrating the Python normalizer; never learn from user/test input. */
const examples = {
  spam: 'Congratulations! Claim your free prize now. Call 07700900123 to collect your reward.',
  ham: 'Are you free for coffee tomorrow? Please call me after work.',
  protected: 'Your free trial is ready. Visit https://example.com/free or call +1 212-555-0199. Email hello@example.com.'
};
const vocabulary = new Set('congratulations claim your free prize now call to collect reward are you for coffee tomorrow please me after work trial is ready visit or email hello meeting see at the win cash urgent reply text stop thanks good home today offer'.split(' '));
const substitutions = { a: '4', e: '€', i: '1', o: '0', s: '$' };
const reverse = { '4': 'a', '€': 'e', '1': 'i', '0': 'o', '$': 's' };
const $ = id => document.getElementById(id);

// Protected spans are tokenized first, so their internal words never change.
function tokenize(text) {
  const pattern = /https?:\/\/[^\s]+|www\.[^\s]+|[\w.+-]+@[\w.-]+\.[a-z]{2,}|\+?\d[\d ()-]{5,}\d|[a-z4€10$]+(?:\.[a-z4€10$]+)*|[^a-z4€10$]/gi;
  return (text.match(pattern) || []).map(value => ({
    value,
    protected: /^(?:https?:\/\/|www\.)/i.test(value) || value.includes('@') || /^\+?\d[\d ()-]{5,}\d$/.test(value),
    word: /^[a-z4€10$]+(?:\.[a-z4€10$]+)*$/i.test(value) && /[a-z]/i.test(value)
  }));
}
function seededRandom(seed) {
  return () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
}
function perturb(text, method, strength, seed = 42) {
  const random = seededRandom(seed);
  const tokens = tokenize(text);
  const eligible = tokens.map((token, index) => ({ token, index })).filter(({ token }) => !token.protected && /^[a-z]{3,}$/i.test(token.value) && (method !== 'substitution' || /[aeios]/i.test(token.value)));
  // Deterministic shuffle chooses an exact proportion of eligible words.
  for (let i = eligible.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [eligible[i], eligible[j]] = [eligible[j], eligible[i]]; }
  const selected = new Set(eligible.slice(0, Math.ceil(eligible.length * strength)).map(item => item.index));
  return tokens.map((token, index) => {
    let output = token.value;
    if (selected.has(index)) {
      if (method === 'substitution') output = output.replace(/[aeios]/i, letter => substitutions[letter.toLowerCase()]);
      if (method === 'punctuation') output = [...output].join('.');
      if (method === 'repetition') output += output.at(-1);
    }
    return { ...token, output, changed: output !== token.value };
  });
}
function normalize(text) {
  const records = [];
  const tokens = tokenize(text).map(token => {
    let output = token.value;
    if (token.protected) records.push(`Protected, unchanged: ${token.value}`);
    else if (token.word && !vocabulary.has(token.value.toLowerCase())) {
      const mapped = token.value.toLowerCase().replace(/[4€10$]/g, char => reverse[char]).replace(/\./g, '');
      let candidates = new Set([mapped]);
      // Remove one duplicate at a time; retain only known, unambiguous words.
      for (let i = 1; i < mapped.length; i++) if (mapped[i] === mapped[i - 1]) candidates.add(mapped.slice(0, i) + mapped.slice(i + 1));
      candidates = [...candidates].filter(word => vocabulary.has(word));
      if (candidates.length === 1) {
        output = candidates[0];
        if (/^[A-Z]/.test(token.value)) output = output[0].toUpperCase() + output.slice(1);
        records.push(`Vocabulary match: ${token.value} → ${output}`);
      } else records.push(`${candidates.length ? 'Ambiguous candidates' : 'No vocabulary match'}, unchanged: ${token.value}`);
    }
    return { ...token, output, changed: output !== token.value };
  });
  return { tokens, records };
}
function renderTokens(id, tokens, normalized = false) {
  const target = $(id);
  target.replaceChildren();
  for (const token of tokens) {
    const node = document.createElement(token.changed ? 'mark' : 'span');
    node.textContent = token.output;
    if (normalized && token.changed) node.className = 'normalized-mark';
    target.append(node);
  }
}
function clearResults() {
  for (const id of ['original-text', 'perturbed-text', 'normalized-text']) $(id).textContent = '—';
  $('change-count').textContent = 'No results yet.';
  const li = document.createElement('li'); li.textContent = 'No results yet.'; $('records').replaceChildren(li);
}
function updateInput() {
  $('count').textContent = `${$('message').value.length} characters`;
  clearResults();
  $('status').textContent = 'Seed: 42. URLs, email addresses and phone numbers are preserved.';
}
function run() {
  const text = $('message').value;
  if (!text.trim()) { clearResults(); $('status').textContent = 'Enter an English message first.'; $('message').focus(); return; }
  const strength = [0.25, 0.5, 1][Number($('strength').value) - 1];
  const perturbed = perturb(text, $('method').value, strength);
  const normalized = normalize(perturbed.map(token => token.output).join(''));
  $('original-text').textContent = text;
  renderTokens('perturbed-text', perturbed);
  renderTokens('normalized-text', normalized.tokens, true);
  const changed = perturbed.filter(token => token.changed).length;
  const restored = normalized.tokens.filter(token => token.changed).length;
  $('change-count').textContent = `${changed} words modified; ${restored} words normalized.`;
  $('status').textContent = changed ? `Modified ${changed} words. Seed: 42.` : 'No eligible English words for this modification. Text unchanged.';
  $('records').replaceChildren(...(normalized.records.length ? normalized.records : ['No changes needed.']).map(record => { const li = document.createElement('li'); li.textContent = record; return li; }));
}
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => { $('message').value = examples[button.dataset.example]; updateInput(); }));
$('message').addEventListener('input', updateInput);
$('method').addEventListener('change', updateInput);
$('strength').addEventListener('input', () => { $('strength-value').textContent = ['25%', '50%', '100%'][Number($('strength').value) - 1]; updateInput(); });
$('generate').addEventListener('click', run);
$('reset').addEventListener('click', () => { $('message').value = examples.spam; $('method').value = 'substitution'; $('strength').value = '2'; $('strength-value').textContent = '50%'; updateInput(); });
$('message').value = examples.spam;
updateInput();
