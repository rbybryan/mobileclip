#!/usr/bin/env bash
#
# Robustly download the HMDB51 evaluation videos and lay them out under
# ``data/hmdb51/videos_test/`` for the zero-shot classification eval.
#
# WHY THIS SCRIPT EXISTS
# ----------------------
# The HuggingFace dataset ``jili5044/hmdb51`` is a SINGLE Xet-backed LFS file
# (``hmdb51.zip``, ~2.12 GB) plus a README. Downloading it through the
# ``hf-mirror.com`` mirror with ``huggingface-cli download`` frequently dies with:
#
#   huggingface_hub.errors.FileMetadataError: Distant resource does not seem to
#   be on huggingface.co. ...
#   huggingface_hub.errors.LocalEntryNotFoundError: ...
#
# Root cause: for Xet-stored files the mirror's HEAD response does not carry the
# ``x-linked-etag`` header, so ``huggingface_hub`` sees ``etag is None`` and
# bails out BEFORE any bytes are fetched. The actual file bytes are perfectly
# reachable -- the ``/resolve/`` URL just 302-redirects to a CDN that speaks
# plain HTTP range requests. So we skip the Python client's metadata check
# entirely and pull the one zip with ``wget -c`` (resumable), then unzip and
# flatten every ``.avi`` into ``videos_test/``.
#
# USAGE
# -----
#   bash tools/download_hmdb51.sh              # uses hf-mirror.com by default
#   HF_ENDPOINT=https://huggingface.co bash tools/download_hmdb51.sh   # official
#
# Idempotent: re-running resumes a partial download and re-verifies the layout.
set -euo pipefail

REPO="datasets/jili5044/hmdb51"
FILE="hmdb51.zip"
EXPECTED_SIZE=2117581749            # bytes, from the HF dataset tree API
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
URL="${ENDPOINT%/}/${REPO}/resolve/main/${FILE}"

RAW_DIR="data/hmdb51_hf"
DEST_DIR="data/hmdb51/videos_test"
ZIP_PATH="${RAW_DIR}/${FILE}"

mkdir -p "${RAW_DIR}" "${DEST_DIR}"

echo ">> Endpoint : ${ENDPOINT}"
echo ">> Source   : ${URL}"
echo ">> Target   : ${ZIP_PATH}"

# 1) Download the single zip, resumable, with retries. -c continues a partial
#    file; --tries/--waitretry ride out flaky mirror connections.
if [ -f "${ZIP_PATH}" ] && [ "$(stat -c%s "${ZIP_PATH}" 2>/dev/null || echo 0)" = "${EXPECTED_SIZE}" ]; then
    echo ">> ${ZIP_PATH} already complete (${EXPECTED_SIZE} bytes), skipping download."
else
    wget -c --tries=10 --waitretry=15 --read-timeout=60 --timeout=60 \
        -O "${ZIP_PATH}" "${URL}"
fi

# 2) Verify the byte count so we don't unzip a truncated / HTML-error file.
ACTUAL_SIZE="$(stat -c%s "${ZIP_PATH}")"
if [ "${ACTUAL_SIZE}" != "${EXPECTED_SIZE}" ]; then
    echo "!! Size mismatch: got ${ACTUAL_SIZE} bytes, expected ${EXPECTED_SIZE}." >&2
    echo "!! Re-run this script to resume the download (wget -c)." >&2
    exit 1
fi
echo ">> Download verified: ${ACTUAL_SIZE} bytes."

# 3) Unzip (skips entries already extracted on a re-run).
echo ">> Unzipping ${ZIP_PATH} ..."
unzip -n -q "${ZIP_PATH}" -d "${RAW_DIR}"

# 4) Flatten every .avi into videos_test/ via symlinks (saves disk). The eval
#    reads bare filenames from anno_downstream/hmdb51_act_test.json, so nested
#    per-class folders must be collapsed to a flat directory.
#
#    Use an ABSOLUTE source root so the symlink targets are absolute paths. A
#    relative target (e.g. from `find data/hmdb51_hf ...`) is resolved relative
#    to the LINK's own directory (data/hmdb51/videos_test/), not the CWD, so
#    every link would dangle -- os.listdir() still lists the names (build passes)
#    but os.path.exists() follows the link and fails (eval reports all missing).
echo ">> Flattening .avi files into ${DEST_DIR} ..."
RAW_ABS="$(cd "${RAW_DIR}" && pwd)"
find "${RAW_ABS}" -name '*.avi' -exec ln -sf {} "${DEST_DIR}/" \;

COUNT="$(find "${DEST_DIR}" -maxdepth 1 -name '*.avi' | wc -l | tr -d ' ')"
echo ">> ${DEST_DIR} now holds ${COUNT} .avi files (expected ~6766)."

# 5) Cross-check against the annotation the eval actually consumes.
ANNO="anno_downstream/hmdb51_act_test.json"
if [ -f "${ANNO}" ]; then
    python - "${DEST_DIR}" "${ANNO}" <<'PY'
import json, os, sys
root, anno = sys.argv[1], sys.argv[2]
data = json.load(open(anno))
miss = [e["video"] for e in data if not os.path.exists(os.path.join(root, e["video"]))]
print(f">> anno check: {len(data)} entries, {len(miss)} missing.")
if miss:
    print(">> examples of missing:", miss[:3])
    print(">> If many are missing it's usually a naming mismatch; any HMDB51 "
          ".avi source works as long as bare filenames match the JSON.")
PY
else
    echo ">> (Skipping anno check: ${ANNO} not found.)"
fi

echo ">> Done."
