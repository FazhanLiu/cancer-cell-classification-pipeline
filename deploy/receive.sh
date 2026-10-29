#!/usr/bin/env bash
# Installed root-owned. Runs as a dedicated deployment user, never as root.
set -Eeuo pipefail
umask 022
environment=${1:?environment required}
case "$environment" in dev) port=8766;; prod) port=8765;; *) exit 64;; esac
[[ $(id -un) == "cancer-deploy-$environment" ]] || exit 77
[[ ${SSH_ORIGINAL_COMMAND:-} =~ ^deploy\ ([0-9a-f]{40})$ ]] || exit 64
release_sha=${BASH_REMATCH[1]}
base="/srv/cancer-$environment"
exec 9>"$base/deploy.lock"
flock -w 900 9
release=$(mktemp -d "$base/releases/${release_sha}.XXXXXXXX")
archive=$(mktemp "$base/releases/.archive.XXXXXXXX")
old=$(readlink "$base/current" || true)
activated=false
cleanup() {
    result=$?
    trap - EXIT
    rm -f "$archive" "$base/current.next"
    if (( result != 0 )); then
        if [[ "$activated" == true ]]; then
            if [[ -n "$old" ]]; then
                ln -s "$old" "$base/current.next"
                mv -Tf "$base/current.next" "$base/current"
                sudo -n /usr/bin/systemctl restart "cancer@$environment.service" || true
            else
                sudo -n /usr/bin/systemctl stop "cancer@$environment.service" || true
                rm -f "$base/current"
            fi
        fi
        rm -rf -- "$release"
    fi
    exit "$result"
}
trap cleanup EXIT
# Bound both transport and expanded size. Only regular files are accepted.
head -c 31457281 > "$archive"
[[ $(stat -c %s "$archive") -le 31457280 ]] || exit 65
python3 /usr/local/libexec/cancer-extract.py "$archive" "$release"
chmod 755 "$release"
[[ -f "$release/requirements-server.txt" && -f "$release/pyproject.toml" ]] || exit 65
# Dependencies are prepared without changing the running release.
python3 -m venv "$release/.venv"
source /etc/cancer/build.env
export PIP_INDEX_URL PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
"$release/.venv/bin/python" -m pip install 'pip<26' 'setuptools>=65' wheel
"$release/.venv/bin/python" -m pip install torch==2.7.0 torchvision==0.22.0 --index-url "$TORCH_INDEX_URL"
"$release/.venv/bin/python" -m pip install -r "$release/requirements-server.txt"
"$release/.venv/bin/python" -m pip install --no-deps --no-build-isolation "$release"
printf 'CANCER_RELEASE=%s\n' "$release_sha" > "$release/release.env"
ln -s "$release" "$base/current.next"
mv -Tf "$base/current.next" "$base/current"
activated=true
sudo -n /usr/bin/systemctl restart "cancer@$environment.service"
for attempt in {1..90}; do
    if curl --silent --fail --max-time 2 "http://127.0.0.1:$port/healthz" | python3 -c 'import json,sys; d=json.load(sys.stdin); sys.exit(d.get("release") != sys.argv[1] or d.get("environment") != sys.argv[2])' "$release_sha" "$environment" 2>/dev/null; then
        echo "Deployed $environment $release_sha"
        exit 0
    fi
    sleep 2
done
echo 'Health check failed; restoring previous release' >&2
exit 1
