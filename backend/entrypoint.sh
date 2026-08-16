#!/bin/sh
# Container entrypoint: fix volume ownership, then drop root before serving.
#
# The image is built to run as the unprivileged `app` user (uid 1000). The
# one thing that user cannot do is take ownership of the Railway volume at
# /app/db, which is created root-owned outside the image. So the container
# starts as root, chowns the mount once (only when its owner is wrong — a
# large uploads/ tree would make an unconditional chown -R slow on every
# boot), and execs the server through setpriv as `app`. util-linux ships
# setpriv in python:*-slim, so no gosu/su-exec dependency.
#
# If the platform already runs the container as a non-root user (e.g.
# Railway's RAILWAY_RUN_UID), there is nothing to drop: exec directly.
set -eu

APP_UID="${APP_UID:-1000}"
APP_GID="${APP_GID:-1000}"

if [ "$(id -u)" = "0" ]; then
    for dir in /app/db /app/data /app/tmp_uploads; do
        [ -d "$dir" ] || continue
        if [ "$(stat -c %u "$dir")" != "$APP_UID" ]; then
            chown -R "$APP_UID:$APP_GID" "$dir"
        fi
    done
    exec setpriv --reuid="$APP_UID" --regid="$APP_GID" --init-groups \
        --inh-caps=-all --no-new-privs "$@"
fi

exec "$@"
