import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const projectRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const workerPath = resolve(projectRoot, "dist/server/index.js");
const manifestPath = resolve(projectRoot, "dist/build-manifest.json");

const [source, manifest] = await Promise.all([
  readFile(workerPath, "utf8"),
  readFile(manifestPath, "utf8"),
]);
const build = JSON.parse(manifest);
assert.equal(build.model_id, '9439f39e3b9e7f45');
assert.equal(build.runtime, 'Cloudflare Workers ESM');
assert.equal(build.worker_bytes, Buffer.byteLength(source));
assert.equal(build.worker_sha256, createHash('sha256').update(source).digest('hex'));

// The source project declares type:module; import the actual generated artifact.
assert.ok(!/^import\s/m.test(source), 'The Worker must not depend on local runtime files.');
const workerModule = await import(pathToFileURL(workerPath));
assert.equal(
  typeof workerModule.default?.fetch,
  "function",
  `${pathToFileURL(workerPath)} must export default.fetch`,
);

const health = await workerModule.default.fetch(new Request('https://demo.example/api/health'));
assert.equal(health.status, 200);
assert.deepEqual(await health.json(), {status:'ok', models_ready:true, model_id:'9439f39e3b9e7f45'});
console.log("Artifact is valid self-contained ESM; frozen models initialize successfully");
