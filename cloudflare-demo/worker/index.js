import { createPredictor } from '../edge/model.mjs';
import { createTextProcessor, METHODS } from '../edge/text.mjs';

const SYSTEMS = ['majority', 'A', 'B', 'C'];
const MAX_BODY_BYTES = 32_768;
const PYTHON_SPACE = /^[\u0009-\u000d\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/u;

function response(body, status = 200, type = 'application/json; charset=utf-8', cache = 'no-store') {
  return new Response(body, { status, headers: {
    'content-type': type,
    'cache-control': cache,
    'x-content-type-options': 'nosniff',
    'referrer-policy': 'strict-origin-when-cross-origin'
  } });
}

function json(value, status = 200) {
  return response(JSON.stringify(value), status);
}

function validate(payload) {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return 'Expected a JSON object.';
  if (Object.keys(payload).some(key => !['text', 'method', 'seed'].includes(key))) return 'Unexpected request field.';
  if (typeof payload.text !== 'string' || Array.from(payload.text).length > 2000 || PYTHON_SPACE.test(payload.text)) {
    return 'Enter a non-empty SMS message of at most 2000 characters.';
  }
  if (!METHODS.includes(payload.method === undefined ? 'substitution' : payload.method)) return 'Choose substitution, punctuation or repetition.';
  if (payload.seed !== undefined && (!Number.isInteger(payload.seed) || payload.seed < 0 || payload.seed > 4294967295)) {
    return 'Seed must be an integer from 0 to 4294967295.';
  }
  return null;
}

/** Initialize fixed model weights during Worker startup, outside request CPU time. */
export function createWorker(loadData, assets) {
  let runtime;
  function models() {
    if (!runtime) {
      const data = loadData();
      runtime = { predictor: createPredictor(data), text: createTextProcessor(data.normalizer) };
    }
    return runtime;
  }

  models();

  return {
    async fetch(request) {
      const path = new URL(request.url).pathname;
      if (path === '/api/health') {
        if (request.method !== 'GET') return json({ detail: 'Method not allowed.' }, 405);
        const { predictor } = models();
        return json({ status: 'ok', models_ready: true, model_id: predictor.modelId });
      }
      if (path === '/api/analyze') {
        if (request.method !== 'POST') return json({ detail: 'Method not allowed.' }, 405);
        if (Number(request.headers.get('content-length')) > MAX_BODY_BYTES) return json({ detail: 'Request is too large.' }, 413);
        let payload;
        try {
          const body = await request.text();
          if (new TextEncoder().encode(body).length > MAX_BODY_BYTES) return json({ detail: 'Request is too large.' }, 413);
          payload = JSON.parse(body);
        } catch {
          return json({ detail: 'Send a valid JSON request.' }, 400);
        }
        const error = validate(payload);
        if (error) return json({ detail: error }, 422);
        const method = payload.method ?? 'substitution';
        const seed = payload.seed ?? 42;
        const { predictor, text } = models();
        const modified = text.perturb(payload.text, method, seed);
        const normalized = text.normalize(modified.text);
        const normalizedOriginal = text.normalize(payload.text).text;
        const predictions = {};
        for (const system of SYSTEMS) {
          predictions[system] = {
            original: predictor.predict(system, system === 'C' ? normalizedOriginal : payload.text),
            modified: predictor.predict(system, system === 'C' ? normalized.text : modified.text),
            threshold: predictor.threshold(system)
          };
        }
        return json({ original_text: payload.text, modified, normalized, predictions, seed, model_id: predictor.modelId });
      }
      if (path.startsWith('/api/')) return json({ detail: 'Not found.' }, 404);
      if (request.method !== 'GET' && request.method !== 'HEAD') return response('Method not allowed.', 405, 'text/plain; charset=utf-8');
      if (path === '/favicon.ico') return new Response(null, { status: 204 });
      const asset = assets[path];
      if (!asset) return response('Not found.', 404, 'text/plain; charset=utf-8');
      return response(request.method === 'HEAD' ? null : asset.body, 200, asset.type, path === '/' ? 'no-cache' : 'public, max-age=3600');
    }
  };
}
