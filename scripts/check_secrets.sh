#!/usr/bin/env bash
# Fail if anything that looks like a credential or an internal path/host is about to be committed.
cd "$(dirname "$0")/.."
pat='(^|[^A-Za-z])sk-[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{12,}|aigc[a-z.-]*\.|/mnt/dolphinfs|/home/hadoop|hadoop-mlm|10\.[0-9]+\.[0-9]+\.[0-9]+|App ?ID|password=|PosterSum|PosterReverse'
if grep -rInE --exclude-dir=.git --exclude-dir=figures --exclude=check_secrets.sh "$pat" . ; then echo "FOUND suspicious strings (see above)"; exit 1; else echo "clean"; fi
