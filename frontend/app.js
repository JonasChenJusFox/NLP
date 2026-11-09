/* Transformations and models share the Python backend's frozen training vocabulary. */
const examples = {
  spam: 'Congratulations! Claim your free prize now. Call 07700900123 to collect your reward.',
  ham: 'Are you free for coffee tomorrow? Please call me after work.',
  protected: 'Your free trial is ready. Visit https://example.com/free or call +1 212-555-0199. Email hello@example.com.'
};
const $ = id => document.getElementById(id);
let pending = null;

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
  const li = document.createElement('li');
  li.textContent = 'No results yet.';
  $('records').replaceChildren(li);
  document.querySelectorAll('[data-prediction]').forEach(cell => { cell.textContent = '—'; });
  $('model-status').textContent = 'Apply a change to compare the trained models.';
}

function updateInput() {
  if (pending) pending.abort();
  pending = null;
  $('generate').disabled = false;
  $('count').textContent = `${$('message').value.length} characters`;
  clearResults();
  $('status').textContent = 'Seed: 42. At most one eligible word changes. URLs, email addresses and numbers are preserved.';
}

async function run() {
  const text = $('message').value;
  if (!text.trim()) {
    clearResults();
    $('status').textContent = 'Enter an English message first.';
    $('message').focus();
    return;
  }
  if (pending) pending.abort();
  const controller = new AbortController();
  pending = controller;
  clearResults();
  $('generate').disabled = true;
  $('status').textContent = 'Running models…';
  try {
    const response = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, method: $('method').value, seed: 42 }),
      signal: controller.signal
    });
    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(typeof error.detail === 'string' ? error.detail : 'The service could not process this message.');
    }
    const result = await response.json();
    if (pending !== controller) return;
    $('original-text').textContent = result.original_text;
    renderTokens('perturbed-text', result.modified.tokens);
    renderTokens('normalized-text', result.normalized.tokens, true);
    const restored = result.normalized.tokens.filter(token => token.changed).length;
    $('change-count').textContent = `${result.modified.changed_word_count} word modified; ${restored} words normalized.`;
    $('status').textContent = result.modified.changed
      ? 'Modified one word. Seed: 42.'
      : 'No eligible English words for this change. Text unchanged.';
    const records = result.normalized.records.map(record => {
      if (record.reason === 'vocabulary_match') return `${record.original} → ${record.output} (training vocabulary)`;
      if (record.reason === 'protected') return `Protected, unchanged: ${record.original}`;
      if (record.reason === 'ambiguous') return `Ambiguous, unchanged: ${record.original} (${record.candidates.join(', ')})`;
      return `No vocabulary match, unchanged: ${record.original}`;
    });
    $('records').replaceChildren(...(records.length ? records : ['No changes needed.']).map(record => {
      const li = document.createElement('li');
      li.textContent = record;
      return li;
    }));
    for (const [system, predictions] of Object.entries(result.predictions)) {
      for (const version of ['original', 'modified']) {
        const prediction = predictions[version];
        $(`${system}-${version}`).textContent = `${prediction.label} (spam score: ${prediction.score.toFixed(3)})`;
      }
    }
    $('model-status').textContent = `Model version: ${result.model_id}. Scores are estimated spam probabilities; each model uses its development-selected threshold.`;
  } catch (error) {
    if (error.name !== 'AbortError' && pending === controller) {
      clearResults();
      $('status').textContent = error instanceof TypeError
        ? 'Cannot reach the backend. Open this demo through the running backend service.'
        : error.message;
    }
  } finally {
    if (pending === controller) {
      pending = null;
      $('generate').disabled = false;
    }
  }
}

document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  $('message').value = examples[button.dataset.example];
  updateInput();
}));
$('message').addEventListener('input', updateInput);
$('method').addEventListener('change', updateInput);
$('generate').addEventListener('click', run);
$('reset').addEventListener('click', () => {
  $('message').value = examples.spam;
  $('method').value = 'substitution';
  updateInput();
});
$('message').value = examples.spam;
updateInput();
