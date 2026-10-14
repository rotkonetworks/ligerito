#!/bin/bash
# Stage the ligerito.rotko.net demo from deploy/ and a parallel-web build
# (scripts/build-wasm-parallel.sh) into OUT, ready for `podman build OUT`.
#
# The wasm, its bindings and the rayon worker helper go under a/<hash>/ and
# are served immutable: every pool worker re-imports them, and Chrome
# serialises concurrent revalidations of one URL, so with no-cache a 32-thread
# pool took one round trip per worker per file to start. index.html stays
# no-cache, so a redeploy is picked up on the next load.

set -euo pipefail

OUT=${1:?usage: stage-site.sh OUT_DIR}
HERE=$(cd "$(dirname "$0")/.." && pwd)
PKG=$HERE/pkg/parallel-web
SRC=$HERE/deploy

H=$(cat "$PKG/ligerito.js" "$PKG/ligerito_bg.wasm" | sha256sum | cut -c1-12)
A=$OUT/a/$H

rm -rf "$OUT"
mkdir -p "$A"
cp "$PKG/ligerito.js" "$PKG/ligerito_bg.wasm" "$A/"
cp -r "$PKG/snippets" "$A/"

# wasm-bindgen-rayon still calls init with positional arguments, which
# wasm-bindgen warns about once per worker
sed -i 's/pkg\.default(data\.module, data\.memory)/pkg.default({ module_or_path: data.module, memory: data.memory })/' \
    "$A"/snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js

sed "s#'\./ligerito\.js'#'./a/$H/ligerito.js'#" "$SRC/index.html" > "$OUT/index.html"
grep -q "a/$H/ligerito.js" "$OUT/index.html"

cp "$SRC"/{nginx.conf,Dockerfile,style.css,site.webmanifest,favicon.ico,favicon-16x16.png,favicon-32x32.png,apple-touch-icon.png,android-chrome-192x192.png,android-chrome-512x512.png} "$OUT/"

echo "staged $OUT (assets a/$H)"
