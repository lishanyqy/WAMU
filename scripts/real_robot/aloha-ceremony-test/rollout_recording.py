"""Timestamped RGB and motor-event recording for ALOHA rollouts."""
import heapq
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


VIDEO_TIME_BASE = Fraction(1, 1_000_000)


class RolloutRecording:
    def __init__(self, root, task):
        self.path = Path(root) / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + task)
        self.path.mkdir(parents=True)
        self.events = (self.path / 'events.jsonl').open('x', buffering=1)
        self.stop = threading.Event()
        self.latest = {}
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.workers = []
        print(f'Recording rollout to {self.path}', flush=True)

    def start(self, pipelines):
        for role, pipeline in pipelines.items():
            self.workers.append(self.pool.submit(self._capture, role, pipeline))

    def _capture(self, role, pipeline):
        previous = None
        stream = None
        with ExitStack() as resources:
            while not self.stop.is_set():
                frames = pipeline.poll_for_frames()
                color = frames.get_color_frame() if frames else None
                if not color or color.get_frame_number() == previous:
                    self.stop.wait(.002)
                    continue
                timestamp = time.monotonic_ns()
                previous = color.get_frame_number()
                rgb = np.asanyarray(color.get_data()).copy()
                self.latest[role] = (previous, rgb)
                if stream is None:
                    origin = timestamp
                    video = resources.enter_context(av.open(str(self.path / f'{role}.mp4'), 'w'))
                    video.metadata['comment'] = f'monotonic_origin_ns={origin}'
                    stream = video.add_stream('libx264', rate=15)
                    stream.height, stream.width = rgb.shape[:2]
                    stream.pix_fmt = 'yuv420p'
                    stream.codec_context.time_base = VIDEO_TIME_BASE
                    stream.options = {'preset': 'ultrafast', 'tune': 'zerolatency', 'crf': '18'}
                frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
                frame.time_base = VIDEO_TIME_BASE
                frame.pts = (timestamp - origin) // 1000
                video.mux(stream.encode(frame))
            if stream is not None:
                video.mux(stream.encode())

    def check(self):
        for worker in self.workers:
            if worker.done():
                worker.result()

    def close(self):
        self.stop.set()
        self.pool.shutdown(wait=True)
        self.events.close()
        self.check()


def camera_frames(path, role):
    import cv2

    movie = path / f'{role}.mp4'
    if movie.exists():
        with av.open(str(movie)) as video:
            origin = int(video.metadata['comment'].removeprefix('monotonic_origin_ns='))
            for frame in video.decode(video=0):
                timestamp = origin + int(frame.pts * frame.time_base * 1_000_000_000)
                yield timestamp, role, frame.to_ndarray(format='bgr24')
    else:
        for filename in sorted((path / role).glob('*.jpg'), key=lambda f: int(f.stem)):
            yield int(filename.stem), role, cv2.imread(str(filename))


def replay(path):
    """Play the three camera views together against their recorded monotonic clock."""
    import cv2

    roles = ['head', 'left_wrist', 'right_wrist']
    views = {role: np.zeros((480, 640, 3), dtype=np.uint8) for role in roles}
    origin = None
    try:
        with ExitStack() as resources:
            sources = [camera_frames(path, role) for role in roles]
            for source in sources:
                resources.callback(source.close)
            for timestamp, role, image in heapq.merge(*sources, key=lambda item: item[0]):
                if origin is None:
                    origin = timestamp
                    start = time.monotonic_ns()
                delay = (timestamp - origin - (time.monotonic_ns() - start)) / 1e9
                if delay > 0:
                    time.sleep(delay)
                if image is not None:
                    cv2.putText(image, role, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
                    views[role] = image
                cv2.imshow('ALOHA episode replay', np.hstack(list(views.values())))
                if cv2.waitKey(1) in (27, ord('q')):
                    break
        if origin is None:
            raise SystemExit('This episode contains no camera frames.')
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Replay an ALOHA episode offline.')
    parser.add_argument('episode', type=Path)
    replay(parser.parse_args().episode)
