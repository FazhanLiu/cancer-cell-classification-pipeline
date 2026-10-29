#!/usr/bin/env bash
# Run once as root on Ubuntu 22.04/24.04. No container images are used.
set -Eeuo pipefail
[[ $EUID == 0 ]] || { echo 'Run with sudo' >&2; exit 77; }
source /etc/os-release
[[ "$ID" == ubuntu ]] || { echo 'This bootstrap supports Ubuntu only' >&2; exit 64; }
mode=${1:-prepare}
case "$mode" in prepare|tls) ;; *) exit 64;; esac
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
if [[ "$mode" == prepare ]]; then
    apt-get update
    apt-get install -y python3 python3-venv nginx apache2-utils certbot curl util-linux
    python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] < (3,13), "Python 3.10–3.12 required"'
    install -d -m 755 /etc/cancer /usr/local/libexec /var/www/cancer-acme
    install -m 755 "$script_dir/receive.sh" /usr/local/libexec/cancer-deploy
    install -m 644 "$script_dir/extract.py" /usr/local/libexec/cancer-extract.py
    install -m 644 "$script_dir/cancer@.service" /etc/systemd/system/cancer@.service
    if [[ ! -f /etc/cancer/build.env ]]; then
        cat > /etc/cancer/build.env <<'BUILD'
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
BUILD
        chmod 644 /etc/cancer/build.env
    fi
fi
for environment in dev prod; do
    if [[ "$environment" == dev ]]; then
        domain=dev.cancercell.otisliu.com; port=8766; memory=25%; cpu=50%
    else
        domain=cancercell.otisliu.com; port=8765; memory=50%; cpu=150%
    fi
    if [[ "$mode" == prepare ]]; then
        id "cancer-$environment" >/dev/null 2>&1 || useradd --system --home-dir "/var/lib/cancer-$environment" --shell /usr/sbin/nologin "cancer-$environment"
        id "cancer-deploy-$environment" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "cancer-deploy-$environment"
        install -d -m 755 "/srv/cancer-$environment"
        chown "cancer-deploy-$environment:cancer-deploy-$environment" "/srv/cancer-$environment"
        install -d -m 755 -o "cancer-deploy-$environment" -g "cancer-deploy-$environment" "/srv/cancer-$environment/releases"
        install -d -m 700 -o "cancer-$environment" -g "cancer-$environment" "/var/lib/cancer-$environment"
        for directory in cases models cache; do
            install -d -m 700 -o "cancer-$environment" -g "cancer-$environment" "/var/lib/cancer-$environment/$directory"
        done
        install -d -m 755 "/etc/systemd/system/cancer@$environment.service.d"
        printf '[Service]\nMemoryMax=%s\nCPUQuota=%s\n' "$memory" "$cpu" > "/etc/systemd/system/cancer@$environment.service.d/limits.conf"
        if [[ ! -f "/etc/cancer/$environment.env" ]]; then
            proxy_token=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
            cat > "/etc/cancer/$environment.env" <<SETTINGS
CANCER_ENV=$environment
CANCER_BIND=127.0.0.1:$port
CANCER_OUTPUT_ROOT=/var/lib/cancer-$environment/cases
CANCER_PROXY_TOKEN=$proxy_token
CANCER_DEVICE=cpu
CANCER_MODEL_PATH=
CANCER_DECISION_PATH=
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-flash
SETTINGS
            chown "root:cancer-$environment" "/etc/cancer/$environment.env"
            chmod 640 "/etc/cancer/$environment.env"
        fi
        # Per-environment deploy accounts can only restart or stop their own unit.
        printf 'cancer-deploy-%s ALL=(root) NOPASSWD: /usr/bin/systemctl restart cancer@%s.service, /usr/bin/systemctl stop cancer@%s.service\n' "$environment" "$environment" "$environment" > "/etc/sudoers.d/cancer-$environment"
        chmod 440 "/etc/sudoers.d/cancer-$environment"
        visudo -cf "/etc/sudoers.d/cancer-$environment"
        # SSH keys go in root-owned files, with a fixed forced command.
        install -d -m 755 /etc/ssh/authorized_keys
        cat > "/etc/ssh/sshd_config.d/cancer-$environment.conf" <<SSH
Match User cancer-deploy-$environment
    AuthorizedKeysFile /etc/ssh/authorized_keys/%u
    AuthenticationMethods publickey
    PasswordAuthentication no
    AllowTcpForwarding no
    X11Forwarding no
    PermitTTY no
    ForceCommand /usr/local/libexec/cancer-deploy $environment
Match all
SSH
    fi
    config="/etc/nginx/sites-available/cancer-$environment"
    if [[ "$mode" == prepare ]]; then
        # Never overwrite a working TLS site during a repeated preparation.
        if [[ ! -e "$config" ]]; then
            cat > "$config" <<HTTP
server {
    listen 80;
    server_name $domain;
    location ^~ /.well-known/acme-challenge/ { root /var/www/cancer-acme; }
    location / { return 503; }
}
HTTP
        fi
    else
        : "${CERTBOT_EMAIL:?Set CERTBOT_EMAIL for certificate registration}"
        certbot certonly --non-interactive --agree-tos --email "$CERTBOT_EMAIL" --webroot -w /var/www/cancer-acme -d "$domain"
        [[ -s "/etc/cancer/$environment.htpasswd" ]] || { echo "Create /etc/cancer/$environment.htpasswd first with htpasswd" >&2; exit 65; }
        proxy_token=$(sed -n 's/^CANCER_PROXY_TOKEN=//p' "/etc/cancer/$environment.env")
        [[ "$proxy_token" =~ ^[a-f0-9]{64}$ ]] || exit 65
        cat > "$config" <<TLS
server {
    listen 80;
    server_name $domain;
    location ^~ /.well-known/acme-challenge/ { root /var/www/cancer-acme; }
    location / { return 301 https://\$host\$request_uri; }
}
server {
    listen 443 ssl;
    server_name $domain;
    ssl_certificate /etc/letsencrypt/live/$domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/$domain/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    auth_basic "Cancer workbench $environment";
    auth_basic_user_file /etc/cancer/$environment.htpasswd;
    client_max_body_size 100m;
    access_log /var/log/nginx/cancer-$environment.access.log;
    error_log /var/log/nginx/cancer-$environment.error.log;
    location = /healthz { return 404; }
    location / {
        proxy_pass http://127.0.0.1:$port;
        proxy_set_header Host $domain;
        proxy_set_header X-Cancer-Proxy-Token $proxy_token;
        proxy_set_header Connection "";
        proxy_http_version 1.1;
        proxy_read_timeout 180s;
    }
}
TLS
        chown root:www-data "$config" "/etc/cancer/$environment.htpasswd"
        chmod 640 "$config" "/etc/cancer/$environment.htpasswd"
    fi
    ln -sfn "$config" "/etc/nginx/sites-enabled/cancer-$environment"
done
if [[ "$mode" == prepare ]]; then
    /usr/sbin/sshd -t
    systemctl reload ssh
    systemctl daemon-reload
    systemctl enable cancer@dev.service cancer@prod.service
else
    install -d -m 755 /etc/letsencrypt/renewal-hooks/deploy
    printf '#!/bin/sh\n/usr/sbin/nginx -t && /usr/bin/systemctl reload nginx\n' > /etc/letsencrypt/renewal-hooks/deploy/cancer-nginx
    chmod 755 /etc/letsencrypt/renewal-hooks/deploy/cancer-nginx
fi
nginx -t
systemctl reload nginx
echo 'Prepared. See deploy/README.md for keys, access credentials and activation.'
