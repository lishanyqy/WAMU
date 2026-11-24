"""Serve the overlay and the RealSense RGB stream on localhost."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
from urllib.parse import urlsplit


def rgb_device(sysfs=Path('/sys/class/video4linux'), devices=Path('/dev')):
    candidates = []
    for node in sorted(sysfs.glob('video*')):
        try:
            interface = (node / 'device/interface').read_text().strip()
            index = (node / 'index').read_text().strip()
        except OSError:
            continue
        if 'RealSense' in interface and interface.endswith(' RGB') and index == '0':
            candidates.append(node)
    top = [
        node for node in candidates
        if (node / 'device/../serial').is_file()
        and (node / 'device/../serial').read_text().strip() == '231622302829'
    ]
    candidates = top or candidates
    if len(candidates) != 1:
        raise RuntimeError('Could not identify a unique RealSense RGB camera.')
    device = devices / candidates[0].name
    if not device.exists():
        raise RuntimeError(f'The RealSense RGB interface was found, but {device} is unavailable.')
    return device


class OverlayHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if urlsplit(path).path == '/overlay.html':
            return str(Path(__file__).with_name('overlay.html'))
        return super().translate_path(path)

    def do_GET(self):
        route = urlsplit(self.path).path
        if route not in ('/camera-status', '/camera.mjpg'):
            return super().do_GET()
        try:
            device = rgb_device()
        except RuntimeError as error:
            self.send_response(503)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(str(error).encode())
            return
        if route == '/camera-status':
            self.send_response(204)
            self.end_headers()
            return

        command = [
            'ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'v4l2',
            '-video_size', '640x480', '-framerate', '15', '-i', str(device),
            '-an', '-c:v', 'mjpeg', '-q:v', '4', '-threads', '1',
            '-f', 'mpjpeg', '-boundary_tag', 'frame', 'pipe:1',
        ]
        with subprocess.Popen(command, stdout=subprocess.PIPE) as capture:
            try:
                chunk = capture.stdout.read1(65536)
                if not chunk:
                    self.send_error(503, 'The RGB camera could not be opened.')
                    return
                self.send_response(200)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.end_headers()
                while chunk:
                    self.wfile.write(chunk)
                    chunk = capture.stdout.read1(65536)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if capture.poll() is None:
                    capture.terminate()


if __name__ == '__main__':
    handler = partial(OverlayHandler, directory=Path(__file__).resolve().parents[4] / "output/first-valid-frames")
    with ThreadingHTTPServer(('127.0.0.1', 0), handler) as server:
        print(f'Open http://127.0.0.1:{server.server_port}/overlay.html', flush=True)
        server.serve_forever()
