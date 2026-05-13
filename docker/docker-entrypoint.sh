#!/bin/sh
# Entrypoint: if running as root, drop privileges to the owner of the
# bind-mounted /work directory before invoking the requested command.
# Dispatch rule:
#   - If the first argument resolves to an executable on PATH (bash, python,
#     obabel, ...), exec it directly.  This lets users run example shell
#     scripts that themselves call htpolynet/obabel.
#   - Otherwise, treat the args as htpolynet subcommand/options.
set -e

# Build the final argv: prepend `htpolynet` unless the user already passed an
# executable as the first arg.
if [ $# -eq 0 ] || ! command -v "$1" >/dev/null 2>&1; then
    set -- htpolynet "$@"
fi

if [ "$(id -u)" -eq 0 ] && [ -d /work ]; then
    WORK_UID=$(stat -c '%u' /work)
    WORK_GID=$(stat -c '%g' /work)
    if [ "$WORK_UID" -ne 0 ]; then
        # gosu sets HOME from the target user's /etc/passwd entry, so create
        # one pointing at /home/htpolynet.  Without this HOME defaults to /
        # and tools (htpolynet, matplotlib, ...) try to write into the
        # read-only image root.
        if ! getent passwd "$WORK_UID" >/dev/null 2>&1; then
            echo "htpolynet:x:$WORK_UID:$WORK_GID:htpolynet:/home/htpolynet:/bin/sh" >> /etc/passwd
        fi
        if ! getent group "$WORK_GID" >/dev/null 2>&1; then
            echo "htpolynet:x:$WORK_GID:" >> /etc/group
        fi
        chown -R "$WORK_UID:$WORK_GID" /home/htpolynet 2>/dev/null || true
        exec gosu "$WORK_UID:$WORK_GID" "$@"
    fi
fi

exec "$@"
