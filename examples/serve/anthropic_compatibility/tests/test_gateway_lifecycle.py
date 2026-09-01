"""End-to-end exercise of the gateway's start_server / stop_server lifecycle.

Everything Slurm-shaped is faked: a stub serve.sh that registers a fleet record
instead of submitting, stub squeue/scancel, and one always-listening backend
that stands in for trtllm-serve.
"""
import http.client, json, os, shutil, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer

GW = "/lustre/fsw/portfolios/coreai/users/serli/workspace/TensorRT-LLM/examples/serve/anthropic_compatibility/gateway.py"
ROOT = tempfile.mkdtemp(prefix="gwtest-")
FLEET = os.path.join(ROOT, "fleet"); os.makedirs(FLEET)
BIN = os.path.join(ROOT, "bin"); os.makedirs(BIN)
LEDGER = os.path.join(ROOT, "ledger")
open(os.path.join(ROOT, "users.txt"), "w").write("serli\n")
open(os.path.join(ROOT, "deploy.yaml"), "w").write("# dummy\n")

failures = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  <- " + str(detail)) if not cond else ""))
    if not cond:
        failures.append(name)

# --- fake backend ----------------------------------------------------------
class Backend(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"data": [{"id": "fake-model"}], "path": self.path,
                           "user": self.headers.get("X-Gateway-User")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    do_POST = do_GET

backend = HTTPServer(("127.0.0.1", 0), Backend)
BPORT = backend.server_address[1]
threading.Thread(target=backend.serve_forever, daemon=True).start()

# --- stub serve.sh / squeue / scancel --------------------------------------
open(os.path.join(BIN, "serve.sh"), "w").write("""#!/usr/bin/env bash
set -u
echo "$*" >> %(ledger)s
case "$1" in
  submit)
    id=$(( $(cat %(root)s/next_id) )); echo $(( id + 1 )) > %(root)s/next_id
    if [[ -f %(root)s/submit_fails ]]; then echo "sbatch: error: refused"; exit 1; fi
    run_dir=%(root)s/run-${id}
    mkdir -p "${run_dir}"
    printf '{"job_id":"%%s","url":"http://127.0.0.1:%(bport)d","run_dir":"%%s","state":"running","end_time":%%s,"heartbeat":%%s}\\n' \\
      "${id}" "${run_dir}" "$(( $(date +%%s) + 14400 ))" "$(date +%%s)" > %(fleet)s/${id}.json
    echo "Submitted batch job ${id}" ;;
  quit)
    run_dir="$2"; id="${run_dir##*-}"
    if [[ -f %(root)s/quit_never_releases ]]; then rm -f %(fleet)s/${id}.json; exit 0; fi
    rm -f %(fleet)s/${id}.json; touch %(root)s/gone-${id} ;;
  *) echo "stub: unknown $1"; exit 2 ;;
esac
""" % dict(ledger=LEDGER, root=ROOT, fleet=FLEET, bport=BPORT))
open(os.path.join(ROOT, "next_id"), "w").write("9000\n")
open(os.path.join(BIN, "squeue"), "w").write(
    '#!/bin/bash\n'
    '# squeue -h -j <id> -o %%T|%%r\n'
    'id="$3"\n'
    'if [[ -f %s/gone-${id} ]]; then exit 0; fi\n'
    'echo "RUNNING|None"\n' % ROOT)
open(os.path.join(BIN, "scancel"), "w").write(
    '#!/bin/bash\necho "scancel $*" >> %s\nrm -f %s/$2.json\ntouch %s/gone-$2\n'
    % (LEDGER, FLEET, ROOT))
for f in ("serve.sh", "squeue", "scancel"):
    os.chmod(os.path.join(BIN, f), 0o755)

# --- run the gateway -------------------------------------------------------
PORT = 18333
env = dict(os.environ, PATH=BIN + os.pathsep + os.environ["PATH"])
log = open(os.path.join(ROOT, "gateway.log"), "w")
proc = subprocess.Popen(
    [sys.executable, GW, "--fleet-dir", FLEET, "--users", os.path.join(ROOT, "users.txt"),
     "--yaml", os.path.join(ROOT, "deploy.yaml"), "--serve-sh", os.path.join(BIN, "serve.sh"),
     "--host", "127.0.0.1", "--port", str(PORT),
     "--discover-interval", "0.3", "--health-interval", "0.3",
     "--supervisor-interval", "0.5", "--release-timeout", "4"],
    stdout=log, stderr=subprocess.STDOUT, env=env)

def req(method, path, timeout=20):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    try:
        conn.request(method, path)
        r = conn.getresponse()
        raw = r.read().decode()
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, raw
    finally:
        conn.close()

for _ in range(100):
    try:
        req("GET", "/_gateway/health"); break
    except OSError:
        time.sleep(0.1)
else:
    sys.exit("gateway never came up:\n" + open(os.path.join(ROOT, "gateway.log")).read())

def wait_for(pred, secs=15):
    end = time.time() + secs
    while time.time() < end:
        s, b = req("GET", "/_gateway/health")
        if pred(b):
            return b
        time.sleep(0.25)
    return b

def ledger():
    return open(LEDGER).read() if os.path.exists(LEDGER) else ""

try:
    print("\n[1] boots idle, holds nothing")
    s, b = req("GET", "/_gateway/health")
    check("health is 200", s == 200, s)
    check("status=stopped", b.get("status") == "stopped", b)
    check("desired=stopped", b.get("desired") == "stopped", b)
    check("names its deployment", b.get("deployment") == "fleet", b)
    check("no serve.sh call at boot", ledger() == "", ledger())

    print("\n[2] /v1 says what to do instead of 'retry shortly'")
    s, b = req("GET", "/v1/models")
    check("503", s == 503, s)
    check("names start_server", "start_server" in json.dumps(b), b)

    print("\n[3] a GET cannot start GPUs")
    s, b = req("GET", "/_gateway/start_server")
    check("405", s == 405, s)
    check("still nothing submitted", ledger() == "", ledger())
    s, b = req("GET", "/_gateway/nope")
    check("unknown path is 404 not 502", s == 404, s)

    print("\n[4] POST start_server submits")
    s, b = req("POST", "/_gateway/start_server")
    check("200", s == 200, s)
    check("action=submitted", b.get("action") == "submitted", b)
    check("job id reported", b.get("job_id") == "9000", b)
    check("serve.sh submit called once", ledger().count("submit") == 1, ledger())
    check("label passed through", "--label start" in ledger(), ledger())

    print("\n[5] it becomes active and proxies")
    b = wait_for(lambda h: h.get("status") == "ok")
    check("status=ok", b.get("status") == "ok", b)
    check("active is the new job", b.get("active") == "9000", b)
    check("pending cleared", b.get("pending") is None, b)
    s, body = req("GET", "/v1/models")
    check("proxied 200", s == 200, s)
    check("reached the backend", body.get("data", [{}])[0].get("id") == "fake-model", body)

    print("\n[6] a second start adopts instead of stacking a job")
    s, b = req("POST", "/_gateway/start_server")
    check("200", s == 200, s)
    check("action=adopted", b.get("action") == "adopted", b)
    check("no second submit", ledger().count("submit") == 1, ledger())

    print("\n[7] stop_server releases through serve.sh quit")
    s, b = req("POST", "/_gateway/stop_server")
    check("200", s == 200, s)
    check("action=stopped", b.get("action") == "stopped", b)
    check("released 9000", b.get("released") == ["9000"], b)
    check("nothing failed", b.get("failed") == [], b)
    check("nothing left allocated", b.get("still_allocated") == [], b)
    check("quit called", "quit " in ledger(), ledger())
    check("scancel NOT used on a registered job", "scancel" not in ledger(), ledger())

    print("\n[8] routing stops immediately, and stays stopped")
    s, b = req("GET", "/_gateway/health")
    check("status=stopped at once", b.get("status") == "stopped", b)
    check("active cleared at once", b.get("active") is None, b)
    s, b = req("GET", "/v1/models")
    check("503 not proxied", s == 503, s)
    time.sleep(2.5)   # several supervisor sweeps
    s, b = req("GET", "/_gateway/health")
    check("still stopped after sweeps", b.get("status") == "stopped", b)
    check("supervisor did not resubmit", ledger().count("submit") == 1, ledger())

    print("\n[9] start again gets a fresh job")
    s, b = req("POST", "/_gateway/start_server")
    check("action=submitted", b.get("action") == "submitted", b)
    check("new job id", b.get("job_id") == "9001", b)
    b = wait_for(lambda h: h.get("status") == "ok")
    check("serving again", b.get("active") == "9001", b)

    print("\n[10] a backend lost outside stop_server is recovered")
    os.remove(os.path.join(FLEET, "9001.json"))
    b = wait_for(lambda h: h.get("active") == "9002", 20)
    check("supervisor resubmitted", b.get("active") == "9002", b)

    print("\n[11] stop, then a failing sbatch backs off instead of storming")
    req("POST", "/_gateway/stop_server")
    open(os.path.join(ROOT, "submit_fails"), "w").close()
    before = ledger().count("submit")
    s, b = req("POST", "/_gateway/start_server")
    check("503 on submit failure", s == 503, s)
    check("action=submit_failed", b.get("action") == "submit_failed", b)
    time.sleep(4)     # ~8 supervisor sweeps at 0.5s
    after = ledger().count("submit") - before
    check("backed off (<=2 attempts in 4s, not 8)", after <= 2, "%d attempts" % after)
    s, fl = req("GET", "/_gateway/fleet")
    check("fleet needs a key", s == 401, s)

    print("\n[12] a job found but not started reads as unmanaged, not ok")
    req("POST", "/_gateway/stop_server")
    os.remove(os.path.join(ROOT, "submit_fails"))
    # A backend that appears while the gateway is stopped: what a restarted
    # gateway sees. It must route to it and say plainly that nobody owns it.
    subprocess.run([os.path.join(BIN, "serve.sh"), "submit", "--yaml", "x",
                    "--label", "byhand"], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    b = wait_for(lambda h: h.get("active") is not None)
    check("routes to it", b.get("active") is not None, b)
    check("status=unmanaged not ok", b.get("status") == "unmanaged", b)
    check("desired still stopped", b.get("desired") == "stopped", b)
    s, body = req("GET", "/v1/models")
    check("still proxies", s == 200, s)
    time.sleep(1.5)
    gwlog = open(os.path.join(ROOT, "gateway.log")).read()
    check("warns in the log", "serving but unmanaged" in gwlog)
    s, b = req("POST", "/_gateway/start_server")
    check("start adopts it", b.get("action") == "adopted", b)
    b = wait_for(lambda h: h.get("status") == "ok")
    check("now managed", b.get("status") == "ok", b)

    print("\n[13] a quit that does not free the allocation is not 'released'")
    open(os.path.join(ROOT, "quit_never_releases"), "w").close()
    active = b.get("active")
    s, b = req("POST", "/_gateway/stop_server", timeout=60)
    check("200 -- the server did stop", s == 200, s)
    check("not claimed as released", b.get("released") == [], b)
    check("reported as still allocated", b.get("still_allocated") == [active], b)
    contains = "scancel" in json.dumps(b)
    check("tells the operator to scancel", contains, b)
    os.remove(os.path.join(ROOT, "quit_never_releases"))

    print("\n[14] gateway survived it all")
    check("process alive", proc.poll() is None, proc.poll())
    check("no traceback in log", "Traceback" not in open(os.path.join(ROOT, "gateway.log")).read())
finally:
    proc.terminate()
    try: proc.wait(5)
    except Exception: proc.kill()
    log.close()

print("\n" + ("=" * 60))
if failures:
    print("FAILED: " + ", ".join(failures))
    print("\n--- gateway log ---\n" + open(os.path.join(ROOT, "gateway.log")).read())
    sys.exit(1)
print("all checks passed")
shutil.rmtree(ROOT, ignore_errors=True)
