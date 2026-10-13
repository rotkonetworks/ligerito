// Cargo target runner for wasm32-wasip1 test binaries, using Node's WASI.
// Preopens the workspace so tests can read test-vectors/.
import { readFile } from 'node:fs/promises';
import { WASI } from 'node:wasi';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';

const workspace = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const [wasmPath, ...rest] = process.argv.slice(2);
const wasi = new WASI({
  version: 'preview1',
  args: [wasmPath, ...rest],
  env: process.env,
  preopens: { [workspace]: workspace },
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exit(wasi.start(instance));
