# Stage camera server (Mac mini)

Copy this folder onto the Mac mini. It only needs Python and OpenCV; MediaPipe
and the rest of osc-gesture run on the Linux desktop.

```bash
cd camera_server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py --camera 0
```

Allow camera access for the terminal/Python when macOS asks. Use `--camera 1`
if the external webcam is not camera 0. The server retries if the camera is
unavailable. Stop with Ctrl-C.

On Linux, from the osc-gesture project directory:

```bash
python main.py --camera-url http://192.168.1.50:8080/stream
```

Replace the IP with the Mac mini's LAN address. The Mac serves the stream;
Linux initiates the connection. Allow incoming TCP port 8080 through the Mac's
firewall. Opening `http://192.168.1.50:8080/` in a browser previews the source.
The server is unauthenticated: use a trusted stage network.

Defaults request 640×480 at 30 FPS, JPEG quality 72. Override with `--width`,
`--height`, `--fps`, and `--quality`. Camera hardware may negotiate different
capture settings. `--host` and `--port` control the server's listening address.
Frames are unmirrored; the Linux tracker performs its usual mirror operation.

Use wired Ethernet between the machines for the stage setup. Capture and JPEG
encoding run independently of HTTP clients, and each client takes the newest
encoded frame. Linux receives continuously into a single-frame slot, skipping
frames when detection is slower than capture. This prevents application queues
from growing; network/socket buffering can still add latency under congestion.

Linux reconnects after connection errors. Frames older than one second since
receipt are discarded, hand state is cleared, and the existing absence/pause
logic runs. This is a receive-time check, not a measurement of camera exposure
age. Manual/mock tabs disconnect Linux from the feed; the standalone Mac server
continues capturing until stopped. The Linux camera picker stays on the remote
source; restart without `--camera-url` to return to local webcams.
