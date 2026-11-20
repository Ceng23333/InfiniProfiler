#!/usr/bin/env bash
set -euo pipefail

# Wrapper for the first Mars integration. Keep HCPTI Profiling API out of the
# continuous path; hcTracer is the nsys-like attach tool.
HCTRACER_BIN="${HCTRACER_BIN:-hcTracer}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
MODE="${MODE:-attach_serve}"
TARGET_PID="${TARGET_PID:-}"
CLIENT_WINDOW_CMD="${CLIENT_WINDOW_CMD:-}"
DURATION_S="${DURATION_S:-10}"

mkdir -p "${OUT_DIR}"

case "${MODE}" in
  smoke)
    cat >"${OUT_DIR}/sample.sh" <<EOF
sleep "${DURATION_S}"
EOF
    chmod +x "${OUT_DIR}/sample.sh"
    (
      cd "${OUT_DIR}"
      "${HCTRACER_BIN}" --hctx --odname raw --name tracer_out ./sample.sh
    )
    ;;
  attach_serve)
    if [[ -z "${TARGET_PID}" || -z "${CLIENT_WINDOW_CMD}" ]]; then
      echo "attach_serve requires TARGET_PID and CLIENT_WINDOW_CMD" >&2
      exit 2
    fi
    mkdir -p "${OUT_DIR}/raw"
    tracer_fifo="${OUT_DIR}/raw/attach_fifo.$$"
    mkfifo "${tracer_fifo}"
    (
      cd "${OUT_DIR}"
      "${HCTRACER_BIN}" --hctx --attach "${TARGET_PID}" \
        --odname raw --name tracer_out <"${tracer_fifo}" \
        >"${OUT_DIR}/raw/attach_tracer.log" 2>&1
    ) &
    tracer_pid=$!
    exec 3>"${tracer_fifo}"
    sleep 2
    client_rc=0
    if kill -0 "${tracer_pid}" 2>/dev/null; then
      bash -c "${CLIENT_WINDOW_CMD}" || client_rc=$?
      # hcTracer documents Ctrl+T as the attach stop signal. Send the literal
      # control byte rather than SIGINT, which can leave capture cleanup behind.
      printf '\024' >&3 || true
      sleep "${ATTACH_STOP_SECS:-5}"
    else
      echo "hcTracer exited before the client window; see raw/attach_tracer.log" >&2
      tail -n 80 "${OUT_DIR}/raw/attach_tracer.log" >&2 || true
      client_rc=1
    fi
    exec 3>&- || true
    wait "${tracer_pid}" 2>/dev/null || true
    rm -f "${tracer_fifo}"
    exit "${client_rc}"
    ;;
  *)
    echo "MODE must be smoke or attach_serve" >&2
    exit 2
    ;;
esac
