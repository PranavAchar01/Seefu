"""Public tunnel for the Twilio phone bridge.

Runs a cloudflared quick tunnel to localhost:8000, writes the assigned host to
runs/.tunnel_host (read by core/phone.py at call time), and stays in the
foreground. Re-run any time; the newest host wins - no server restart needed.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNNEL_FILE = ROOT / "runs/.tunnel_host"


def main():
    TUNNEL_FILE.parent.mkdir(exist_ok=True)
    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    host = None
    try:
        for line in proc.stdout:
            m = re.search(r"https://([a-z0-9-]+\.trycloudflare\.com)", line)
            if m and host is None:
                host = m.group(1)
                TUNNEL_FILE.write_text(host + "\n")
                print(f"\ntunnel LIVE: https://{host}  (written to {TUNNEL_FILE})")
                print("leave this running; phone calls route through it\n")
            if "error" in line.lower():
                print(line.rstrip())
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        TUNNEL_FILE.unlink(missing_ok=True)
        print("tunnel closed, host file removed")
        sys.exit(0)


if __name__ == "__main__":
    main()
