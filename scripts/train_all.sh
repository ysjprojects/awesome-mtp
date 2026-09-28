#!/usr/bin/env bash
# Train the four reference models used in the tutorials (5-9 minutes each on an M1 Pro / MPS).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
STEPS=${STEPS:-1200}
COMMON=(--steps "$STEPS" --eval-every 300 --log-every 100)

$PY -m mtp.train --kind none                                        --out runs/ntp        "${COMMON[@]}"
$PY -m mtp.train --kind parallel   --n-future 3                     --out runs/parallel3  "${COMMON[@]}"
$PY -m mtp.train --kind sequential --n-future 2                     --out runs/seq2       "${COMMON[@]}"
$PY -m mtp.train --kind sequential --n-future 3 --share-weights     --out runs/shared3    "${COMMON[@]}"
