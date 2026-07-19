#!/bin/sh
# Kubeflow's notebook-controller normally injects NB_PREFIX=/notebook/<ns>/<name>.
# If it's missing but we're in a k8s pod, derive it: the namespace comes from the
# serviceaccount mount and a notebook StatefulSet pod is named <server-name>-<ordinal>.
NS_FILE=/var/run/secrets/kubernetes.io/serviceaccount/namespace
if [ -z "$NB_PREFIX" ] && [ -f "$NS_FILE" ]; then
    ns=$(cat "$NS_FILE")
    case "$HOSTNAME" in
        *-[0-9]|*-[0-9][0-9])
            export NB_PREFIX="/notebook/$ns/${HOSTNAME%-*}"
            ;;
    esac
fi
# normalize: no trailing slashes, so base_url never gets a double slash
while [ "${NB_PREFIX%/}" != "$NB_PREFIX" ]; do NB_PREFIX="${NB_PREFIX%/}"; done
export NB_PREFIX="${NB_PREFIX:-}"

# If the cluster's Istio VirtualService strips the /notebook/<ns>/<name> prefix,
# requests arrive as /jupyter/... while Jupyter's base_url includes the prefix.
# Re-add it in nginx so both stripped and unstripped clusters work.
if [ -n "$NB_PREFIX" ]; then
    cat > /etc/nginx/nb-prefix/reprefix.conf <<EOF
location ~ ^/jupyter(/.*)?\$ {
    rewrite ^/jupyter(/.*)?\$ ${NB_PREFIX}/jupyter\$1 last;
}
EOF
fi

# expose runtime facts next to the build stamp (served at <prefix>/build-info)
{
    head -1 /app/build-info 2>/dev/null
    echo "NB_PREFIX=${NB_PREFIX}"
    echo "HOSTNAME=${HOSTNAME}"
    echo "uid=$(id -u) gid=$(id -g)"
    echo "notebooks_dir: $(ls -ld /app/notebooks 2>&1)"
} > /app/build-info.tmp && mv /app/build-info.tmp /app/build-info

exec supervisord -c /etc/supervisor/supervisord.conf
