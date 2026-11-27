#!/usr/bin/env bash
# Fetches the NVIDIA files DLSS and DLSS frame generation need, for VIctory-NVIDIA's Streamline code
# (crates/victory-gpu/src/streamline.rs, VICTORY_STREAMLINE_DIR). Our own replacement for upstream's
# Get-NVIDIA-DLSS.cmd / get-nvidia-sdks.ps1, stricter:
#   - one source only: NVIDIA's Streamline release on github.com/NVIDIA-RTX/Streamline, pinned tag;
#   - the zip's SHA-256 checked against the digest GitHub publishes for the release asset (stops if absent
#     or different);
#   - only the 8 DLLs + licence texts are taken out; every DLL must carry a valid Authenticode signature
#     from NVIDIA Corporation, or nothing is installed;
#   - nothing downloaded is ever run. Writes only under %LOCALAPPDATA%\victory-nvidia\.
# Usage: bash get-dlss.sh            (tag v2.14.16, the one VIctory's code was written against)
set -euo pipefail

TAG="${1:-v2.14.16}"
REPO="NVIDIA-RTX/Streamline"
ASSET="streamline-sdk-$TAG.zip"
BASE="$(cygpath -u "${LOCALAPPDATA:?run from Git Bash on Windows}")/victory-nvidia"
WORK="$BASE/dl-streamline-$TAG"          # its own new, empty folder for the untrusted download
DEST="$BASE/nvidia-sdk/Streamline/bin/x64"
DLLS=(sl.interposer.dll sl.common.dll sl.dlss.dll sl.dlss_g.dll sl.reflex.dll sl.pcl.dll nvngx_dlss.dll nvngx_dlssg.dll)
TAR=/c/Windows/System32/tar.exe         # Windows' bsdtar reads zips

die() { echo "STOPPED: $*" >&2; exit 1; }

rm -rf "$WORK"; mkdir -p "$WORK"

echo "1/5 asking GitHub for $REPO $TAG's asset $ASSET and its published SHA-256"
curl -fsSL -H "Accept: application/vnd.github+json" -H "User-Agent: victory-get-dlss" \
  "https://api.github.com/repos/$REPO/releases/tags/$TAG" -o "$WORK/release.json"
read -r SIZE DIGEST URL < <(python -I -c '
import json,sys
r=json.load(sys.stdin)
a=[x for x in r.get("assets",[]) if x["name"]==sys.argv[1]]
if not a: sys.exit("no asset")
a=a[0]; print(a["size"], (a.get("digest") or "none"), a["browser_download_url"])
' "$ASSET" < "$WORK/release.json" | tr -d '\r') || die "release $TAG has no $ASSET"
case "$URL" in https://github.com/$REPO/releases/download/*) ;; *) die "unexpected download URL $URL";; esac
[[ "$DIGEST" == sha256:* ]] || die "GitHub publishes no SHA-256 for $ASSET (digest=$DIGEST); not downloading unchecked"
WANT="${DIGEST#sha256:}"
echo "    $ASSET: $((SIZE / 1048576)) MB, sha256 $WANT"

echo "2/5 downloading $URL"
curl -fL --retry 3 -H "User-Agent: victory-get-dlss" -o "$WORK/$ASSET" "$URL"
GOT=$(sha256sum "$WORK/$ASSET" | cut -d' ' -f1)
[[ "$GOT" == "$WANT" ]] || die "SHA-256 mismatch: got $GOT, GitHub says $WANT"
echo "    SHA-256 matches"

echo "3/5 taking out the 8 DLLs and the licences (release bin\\x64, not development)"
LIST=$("$TAR" -tf "$WORK/$ASSET")
PICK=()
for d in "${DLLS[@]}"; do
  p=$(grep -E "(^|/)bin/x64/$d$" <<<"$LIST" | grep -v "/development/" | head -1) || true
  [[ -n "$p" ]] || die "$d not in the archive"
  PICK+=("$p")
done
while IFS= read -r p; do PICK+=("$p"); done < <(grep -iE "(^|/)bin/x64/[^/]*licen[sc]e[^/]*\.txt$|^([^/]+/)?license\.txt$" <<<"$LIST" | grep -v "/development/")
mkdir -p "$WORK/x"
"$TAR" -xf "$WORK/$ASSET" -C "$WORK/x" "${PICK[@]}"

echo "4/5 checking every DLL's Authenticode signature (must be valid and NVIDIA Corporation's)"
for d in "${DLLS[@]}"; do
  f=$(find "$WORK/x" -path "*/bin/x64/$d" | grep -v /development/ | head -1)
  w=$(cygpath -w "$f")
  out=$(powershell -NoProfile -Command "\$s = Get-AuthenticodeSignature -LiteralPath '$w'; '{0}|{1}' -f \$s.Status, \$s.SignerCertificate.Subject")
  status=${out%%|*}; subject=${out#*|}
  [[ "$status" == Valid* ]] || die "$d signature status: $status"
  [[ "$subject" == *"O=NVIDIA Corporation"* ]] || die "$d is signed by someone else: $subject"
  echo "    $d: valid, NVIDIA Corporation"
done

echo "5/5 installing into $DEST"
mkdir -p "$DEST"
for d in "${DLLS[@]}"; do cp "$(find "$WORK/x" -path "*/bin/x64/$d" | grep -v /development/ | head -1)" "$DEST/"; done
find "$WORK/x" -iname "*licen*e*.txt" -exec cp {} "$DEST/" \;
rm -rf "$WORK"
echo
echo "Done. NVIDIA's files under NVIDIA's licences (saved beside them) in:"
echo "  $DEST"
echo "Point the emulator at them with: export VICTORY_STREAMLINE_DIR='$(cygpath -w "$DEST")'"
