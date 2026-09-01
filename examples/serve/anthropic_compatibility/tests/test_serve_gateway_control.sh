#!/usr/bin/env bash
# Exercise serve.sh's gateway_control() in isolation, against a stub gateway.
set -uo pipefail
SERVE=/lustre/fsw/portfolios/coreai/users/serli/workspace/TensorRT-LLM/examples/serve/anthropic_compatibility/serve.sh
ROOT=$(mktemp -d /tmp/servetest-XXXX)
fails=0
check() { if [[ "$2" == "$3" ]]; then echo "  PASS  $1"; else echo "  FAIL  $1"; echo "        want: $3"; echo "        got : $2"; fails=$((fails+1)); fi; }
contains() { if [[ "$2" == *"$3"* ]]; then echo "  PASS  $1"; else echo "  FAIL  $1"; echo "        want substring: $3"; echo "        got : $2"; fails=$((fails+1)); fi; }

# stub gateway: status code and body come from files, so each case can vary them
cat > "${ROOT}/stub.py" <<'PY'
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
root = sys.argv[1]
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def respond(self):
        health = self.path.startswith("/_gateway/health")
        suffix = "_health" if health and self.command == "GET" else ""
        code = int(open(root + "/code" + suffix).read().strip())
        body = open(root + "/body" + suffix, "rb").read()
        open(root + "/hits", "a").write("%s %s\n" % (self.command, self.path))
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    do_GET = do_POST = respond
s = HTTPServer(("127.0.0.1", int(sys.argv[2])), H)
print("up", flush=True)
s.serve_forever()
PY

PORT=18999
echo 200 > "${ROOT}/code"; printf '{"action": "submitted"}' > "${ROOT}/body"
echo 200 > "${ROOT}/code_health"
printf '{\n  "status": "ok",\n  "deployment": "computelab_glm5.2"\n}' > "${ROOT}/body_health"
python3 "${ROOT}/stub.py" "${ROOT}" "${PORT}" > "${ROOT}/stub.log" 2>&1 &
STUB=$!
for _ in $(seq 50); do grep -q up "${ROOT}/stub.log" 2>/dev/null && break; sleep 0.1; done
trap 'kill ${STUB} 2>/dev/null; rm -rf "${ROOT}"' EXIT

# gateway_control() lifted verbatim, with only the two things main() would supply.
harness() {
    local script="${ROOT}/harness.sh"
    {
        echo 'set -euo pipefail'
        echo 'die() { echo "serve.sh: $*" >&2; exit 1; }'
        sed -n '/^gateway_curl()/,/^}/p' "${SERVE}"
        sed -n '/^gateway_control()/,/^}/p' "${SERVE}"
        echo "CFG_FLEET_DIR='${ROOT}/fleet'"
        echo "ARG_YAML='/some/deploy.yaml'"
        echo "CFG_NAME='computelab_glm5.2'"
        echo 'gateway_control "$1"'
    } > "${script}"
    bash "${script}" "$1" 2>&1
}

mkdir -p "${ROOT}/fleet"

echo "[1] no gateway_url yet"
out=$(harness start); rc=$?
check "exit 1" "${rc}" "1"
contains "explains how to start one" "${out}" "--submit"

echo "[2] happy path: 200"
echo "http://127.0.0.1:${PORT}" > "${ROOT}/fleet/gateway_url"
out=$(harness start); rc=$?
check "exit 0" "${rc}" "0"
check "prints the JSON body" "${out}" '{"action": "submitted"}'
contains "POSTed to start_server" "$(cat "${ROOT}/hits")" "POST /_gateway/start_server"

echo "[3] --stop POSTs stop_server, --status GETs health"
: > "${ROOT}/hits"
harness stop >/dev/null; harness status >/dev/null
contains "stop is a POST" "$(cat "${ROOT}/hits")" "POST /_gateway/stop_server"
contains "status is a GET" "$(cat "${ROOT}/hits")" "GET /_gateway/health"

echo "[4] non-2xx still shows the body, then fails"
echo 503 > "${ROOT}/code"; printf '{"action": "submit_failed", "message": "sbatch refused"}' > "${ROOT}/body"
out=$(harness start); rc=$?
check "exit 1" "${rc}" "1"
contains "body shown before dying" "${out}" "sbatch refused"
contains "names the status" "${out}" "HTTP 503"

echo "[5] empty body does not confuse the status split"
echo 200 > "${ROOT}/code_health"; : > "${ROOT}/body_health"
out=$(harness status); rc=$?
check "exit 0" "${rc}" "0"
check "prints nothing but a newline" "${out}" ""

echo "[6] body that ends in a newline"
echo 200 > "${ROOT}/code_health"; printf '{"status": "ok"}\n' > "${ROOT}/body_health"
out=$(harness status); rc=$?
check "exit 0" "${rc}" "0"
check "body intact" "${out}" '{"status": "ok"}'

echo "[7] multi-line body (json_response uses indent=2)"
printf '{\n  "status": "ok",\n  "desired": "running"\n}' > "${ROOT}/body_health"
out=$(harness status); rc=$?
check "exit 0" "${rc}" "0"
contains "keeps every line" "${out}" '"desired": "running"'
contains "keeps the first line" "${out}" '{'

printf '{"status": "ok", "deployment": "computelab_glm5.2"}' > "${ROOT}/body_health"

echo "[8] gateway unreachable"
kill ${STUB} 2>/dev/null; wait ${STUB} 2>/dev/null
out=$(harness start); rc=$?
check "exit 1" "${rc}" "1"
contains "says it cannot reach it" "${out}" "cannot reach the gateway"
contains "names the stale file" "${out}" "gateway_url"
contains "suggests squeue" "${out}" "squeue"

echo "[9] the preflight refuses a gateway_url pointing at another deployment"
python3 "${ROOT}/stub.py" "${ROOT}" "${PORT}" > "${ROOT}/stub.log" 2>&1 &
STUB=$!
for _ in $(seq 50); do grep -q up "${ROOT}/stub.log" 2>/dev/null && break; sleep 0.1; done
echo "http://127.0.0.1:${PORT}" > "${ROOT}/fleet/gateway_url"
printf '{"status": "ok", "deployment": "computelab_deepseek_v4"}' > "${ROOT}/body_health"
: > "${ROOT}/hits"
out=$(harness stop); rc=$?
check "exit 1" "${rc}" "1"
contains "names the gateway it found" "${out}" "computelab_deepseek_v4"
contains "names the one it wanted" "${out}" "computelab_glm5.2"
contains "says it is refusing" "${out}" "Refusing to stop"
check "no POST was sent" "$(grep -c POST "${ROOT}/hits" || true)" "0"

echo "[10] --status skips the preflight (read-only, and useful when stale)"
out=$(harness status); rc=$?
check "exit 0" "${rc}" "0"
contains "shows whatever is there" "${out}" "computelab_deepseek_v4"

echo "[11] the preflight refuses when nothing answers /_gateway/health"
printf '{"status": "ok", "deployment": "computelab_glm5.2"}' > "${ROOT}/body_health"
echo 500 > "${ROOT}/code_health"
out=$(harness start); rc=$?
check "exit 1" "${rc}" "1"
contains "says health did not answer" "${out}" "/_gateway/health"
echo 200 > "${ROOT}/code_health"

echo "[12] control flags are rejected by the other commands"
cat > "${ROOT}/reject.sh" <<EOF
set -euo pipefail
die() { echo "serve.sh: \$*" >&2; exit 1; }
$(sed -n '/^reject_control_flags()/,/^}/p' "${SERVE}")
ARG_CONTROL="stop"; ARG_YAML="/some/deploy.yaml"
reject_control_flags
EOF
out=$(bash "${ROOT}/reject.sh" 2>&1); rc=$?
check "exit 1" "${rc}" "1"
contains "points at the right command" "${out}" "serve.sh gateway --yaml /some/deploy.yaml --stop"
grep -q 'reject_control_flags' <(sed -n '/^cmd_submit()/,/^}/p' "${SERVE}") \
  && echo "  PASS  cmd_submit calls it" || { echo "  FAIL  cmd_submit calls it"; fails=$((fails+1)); }
grep -q 'reject_control_flags' <(sed -n '/^cmd_run()/,/^}/p' "${SERVE}") \
  && echo "  PASS  cmd_run calls it" || { echo "  FAIL  cmd_run calls it"; fails=$((fails+1)); }

echo "[13] empty gateway_url file"
: > "${ROOT}/fleet/gateway_url"
out=$(harness start); rc=$?
check "exit 1" "${rc}" "1"
contains "explicit about the empty file" "${out}" "empty gateway address"

echo
if (( fails )); then echo "FAILED: ${fails} check(s)"; exit 1; fi
echo "all serve.sh checks passed"
