"""Run the supervised ALOHA startup and a timed WAMU trial at a configurable action rate."""
import argparse
import io
import json
import math
import os
import signal
import subprocess
import threading
import sys
from contextlib import ExitStack, contextmanager
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WAMU_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WAMU_ROOT / 'WAMU'))

import lerobot
import numpy as np
import pyrealsense2 as rs
from serial import SerialException
import torch
from PIL import Image
from rollout_recording import RolloutRecording
from websockets.sync.client import connect

from benchmarks.utils.client import build_payload
from lerobot.common.robot_devices.inference_pose import AlohaInferenceMotion
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.utils.utils import init_hydra_config

OUT = WAMU_ROOT / 'output/aloha-ceremony-test'
OUT.mkdir(parents=True, exist_ok=True)
REAL_RESULTS = WAMU_ROOT / 'real_results'
LEROBOT_ROOT = Path(lerobot.__file__).resolve().parents[1]
MODELS = (
    'wamu_real_self_attn',
    'wamu_real_pretrain_robot_self_attn',
    'wamu_real_pretrain_foundation',
)
WEIGHT_STEPS = (12000, 15000, 20000)
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8848
RIG = Path('/home/King/Desktop/QuestController/webxr_teleop/hardware/rig.json')
JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
SERIALS = dict(head='231622302829', left_wrist='218622270994', right_wrist='218622274350')
TRAINING_PROMPTS = {
    'bowls': 'Sort the blocks by color, placing the green blocks in the left bowl and the red blocks in the right bowl.',
    'cupstack': 'Stack the three cups into a single stack.',
    'eggpack': 'Place the loose eggs into the empty slots of the egg carton.',
    'headphones': 'Remove the headphones from the stand and place them on the table.',
}
DEFAULT_ROLLOUT_HZ = 15.0
SERVO_HZ = 45.0
HORIZON = 8
CAMERA_STARTUP_TIMEOUT_S = 10
ROLLOUT_RGB_TIMEOUT_S = 1.0


class CameraStallError(RuntimeError):
    """A rollout observation waited too long for fresh camera frames."""


def open_camera(role, serial):
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
    for attempt in range(2):
        pipeline = rs.pipeline()
        profile = None
        try:
            profile = pipeline.start(config)
            try:
                frames = pipeline.wait_for_frames(CAMERA_STARTUP_TIMEOUT_S * 1000)
                if not frames.get_color_frame():
                    raise RuntimeError('First frameset did not contain RGB')
            except BaseException:
                try:
                    pipeline.stop()
                except RuntimeError as stop_error:
                    print(f'{role} ({serial}) RGB cleanup failed: {stop_error}', file=sys.stderr, flush=True)
                raise
        except RuntimeError as exc:
            if attempt == 1:
                raise RuntimeError(f'{role} ({serial}) failed to start RGB after retry: {exc}') from exc
            print(f'{role} ({serial}) RGB startup failed: {exc}. Retrying once.', flush=True)
            if profile is not None:
                print(f'Resetting the {role} RealSense before reopening RGB.', flush=True)
                try:
                    profile.get_device().hardware_reset()
                except RuntimeError as reset_error:
                    raise RuntimeError(f'{role} ({serial}) RealSense reset failed: {reset_error}') from reset_error
            del profile, pipeline
            time.sleep(2)
            context = rs.context()
            deadline = time.monotonic() + CAMERA_STARTUP_TIMEOUT_S
            while not any(device.get_info(rs.camera_info.serial_number) == serial
                          for device in context.query_devices()):
                if time.monotonic() >= deadline:
                    raise RuntimeError(f'{role} ({serial}) did not reappear after RGB startup failed') from exc
                time.sleep(.25)
        else:
            return pipeline


def find_checkpoint(model, step):
    family = WAMU_ROOT / 'models' / 'action_film' / model
    candidates = sorted(p for p in family.glob(f'*/checkpoint_step_{step}.safetensors') if p.is_file())
    if not candidates:
        raise ValueError(f'No local {model} checkpoint at step {step} under {family}.')
    if len(candidates) != 1:
        raise ValueError(f'Multiple local {model} checkpoints at step {step}: {candidates}')
    return candidates[0].resolve()


def serve_checkpoint(checkpoint, horizon=HORIZON, inference_mode='sync'):
    python = Path('/home/King/miniconda3/envs/openwam/bin/python')
    repo = WAMU_ROOT / 'WAMU'
    os.chdir(repo)
    os.execv(str(python), [
        str(python), str(repo / 'scripts' / 'deploy.py'),
        '--ckpt-dir', str(checkpoint.parent), '--ckpt-name', checkpoint.name,
        '--device', 'cuda:0', '--host', SERVER_HOST, '--port', str(SERVER_PORT),
        '--compile-enabled', 'false', '--inference-mode', inference_mode, '--inference-horizon', str(horizon),
    ])


def verify_checkpoint(ws, checkpoint, horizon=HORIZON, inference_mode='sync'):
    ws.send('{"type":"ping"}')
    reply = json.loads(ws.recv(timeout=2))
    if reply.get('type') != 'pong' or reply.get('checkpoint') != str(checkpoint):
        actual = reply.get('checkpoint', 'checkpoint identity unavailable')
        raise RuntimeError(
            f'Expected {checkpoint}; server reports {actual}. '
            'Launch the selected checkpoint with --serve-only before running hardware.'
        )
    expected = dict(mode=inference_mode, inference_horizon=horizon)
    execution = reply.get('execution', {})
    if any(execution.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f'Expected WAMU execution settings {expected}; server reports {execution}.')


@contextmanager
def selected_server(checkpoint, model, weights, horizon=HORIZON, inference_mode='sync'):
    import psutil

    url = f'ws://{SERVER_HOST}:{SERVER_PORT}'
    try:
        with connect(url, open_timeout=2, close_timeout=.2, proxy=None) as ws:
            verify_checkpoint(ws, checkpoint, horizon, inference_mode)
    except ConnectionRefusedError:
        pass
    except RuntimeError:
        # Replace only an idle local WAMU deploy process belonging to this user.
        listeners = [c for c in psutil.net_connections('tcp') if c.status == 'LISTEN'
                     and c.laddr.port == SERVER_PORT and c.laddr.ip == SERVER_HOST]
        if len(listeners) != 1 or listeners[0].pid is None:
            raise RuntimeError('Cannot identify the existing WAMU listener.')
        proc = psutil.Process(listeners[0].pid)
        script = str(WAMU_ROOT / 'WAMU/scripts/deploy.py')
        if proc.uids().effective != os.geteuid() or script not in proc.cmdline():
            raise RuntimeError('Port 8848 is occupied by another application.')
        if any(c.status == 'ESTABLISHED' for c in proc.net_connections('tcp')):
            raise RuntimeError('The previous WAMU server still has connected clients.')
        proc.terminate()
        proc.wait(timeout=30)
    else:
        yield
        return

    child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--serve-only',
                              '--model', model, '--weights', str(weights),
                              '--horizon', str(horizon), '--inference-mode', inference_mode], start_new_session=True)
    try:
        deadline = time.monotonic() + 300
        while child.poll() is None:
            try:
                with connect(url, open_timeout=2, close_timeout=.2, proxy=None) as ws:
                    verify_checkpoint(ws, checkpoint, horizon, inference_mode)
                break
            except ConnectionRefusedError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('WAMU startup exceeded five minutes.')
                time.sleep(.5)
        else:
            raise RuntimeError(f'WAMU server exited during startup ({child.returncode}).')
        yield
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=30)


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError('duration must be a finite, positive number of seconds')
    return seconds


def nonnegative_speed(value):
    speed = float(value)
    if not math.isfinite(speed) or speed < 0:
        raise argparse.ArgumentTypeError('speed must be finite and nonnegative (0 disables rollout slew limiting)')
    return speed


def follower_power(side, bus, *, allow_off=False):
    faults = bus.read('Hardware_Error_Status')
    if faults.any():
        details = {n: int(v) for n, v in zip(bus.motor_names, faults, strict=True) if v}
        raise RuntimeError(f'{side}: hardware error registers {details}')
    torque = bus.read('Torque_Enable')
    if (torque == 1).all():
        return True
    if allow_off and (torque == 0).all():
        return False
    raise RuntimeError(f'{side}: unexpected Torque_Enable registers {torque.tolist()}')


def read_startup_positions(robot, calibrations, read, record):
    for attempt in range(2):
        try:
            for side, bus in robot.follower_arms.items():
                bus.connect()
                bus.set_calibration(calibrations[side][0])
            robot.is_connected = True
            for side, bus in robot.follower_arms.items():
                follower_power(side, bus, allow_off=True)
            return read()
        except SerialException as exc:
            if attempt:
                raise
            record('startup_serial_retry', message=str(exc))
            print(f'Follower serial communication failed during startup: {exc}. '
                  'Reopening follower ports once before motion.', flush=True)
            for bus in robot.follower_arms.values():
                if bus.is_connected:
                    bus.disconnect()
            robot.is_connected = False
            deadline = time.monotonic() + 5
            time.sleep(.5)
            while not all(Path(bus.port).exists() for bus in robot.follower_arms.values()):
                if time.monotonic() >= deadline:
                    missing = [bus.port for bus in robot.follower_arms.values() if not Path(bus.port).exists()]
                    raise RuntimeError(f'Follower serial ports unavailable after startup failure: {missing}') from exc
                time.sleep(.1)


def establish_startup_hold(arms):
    # Prepare both followers before powering either one. Raw encoder ticks
    # preserve the current position through calibration and quantization.
    pending = {}
    for side, bus in arms.items():
        if follower_power(side, bus, allow_off=True):
            continue
        modes = bus.read('Operating_Mode')
        expected = [5 if name == 'gripper' else 4 for name in bus.motor_names]
        if modes.tolist() != expected or bus.read('Bus_Watchdog').any():
            raise RuntimeError(f'{side}: unexpected operating mode or active watchdog')
        calibration = bus.calibration
        bus.set_calibration(None)
        try:
            if (np.abs(bus.read('Present_Velocity').astype(np.int32)) > 2).any():
                raise RuntimeError(f'{side}: follower moving before torque enable')
            current = bus.read('Present_Position').copy()
            bus.write('Goal_Position', current.copy())
            tolerances = [5 if name == 'gripper' else 2 for name in bus.motor_names]
            goal = bus.read('Goal_Position').astype(np.int64)
            delta = goal - current
            mismatches = [
                f'{name} (present={present}, goal={readback}, delta={diff}, tolerance={tolerance} ticks)'
                for name, present, readback, diff, tolerance in zip(
                    bus.motor_names, current, goal, delta, tolerances, strict=True
                )
                if abs(diff) > tolerance
            ]
            if mismatches:
                raise RuntimeError(f'{side}: measured hold goal readback mismatch: {", ".join(mismatches)}')
            pending[side] = current
        finally:
            bus.set_calibration(calibration)
    for side, current in pending.items():
        bus = arms[side]
        calibration = bus.calibration
        bus.set_calibration(None)
        try:
            if (np.abs(bus.read('Present_Position').astype(np.int64) - current) > 2).any():
                raise RuntimeError(f'{side}: position changed before torque enable')
            bus.write('Torque_Enable', 1)
            follower_power(side, bus)
        finally:
            bus.set_calibration(calibration)


def opening_ceremony(motion):
    # Match the sleep.py CLI trajectory: five seconds of linear interpolation at 50 Hz.
    start = motion._positions()
    target = motion.ceremony_target
    if motion._send(start) is None:
        return False
    begun = time.monotonic()
    speed = (target - start).abs() / 5
    for step in range(1, 251):
        time.sleep(max(0, begun + step / 50 - time.monotonic()))
        if motion._send(torch.lerp(start, target, step / 250), speed=speed) is None:
            return False
    return motion._settle(target)


def send_one_follower(robot, send, target, active_side, held_position):
    followers = robot.follower_arms
    active_slice = slice(9, 18) if active_side == 'right' else slice(0, 9)
    try:
        # Reuse LeRobot's normal relative target limiting and motor write.
        robot.follower_arms = {active_side: followers[active_side]}
        sent = send(target[active_slice])
    finally:
        robot.follower_arms = followers
    return torch.cat((held_position, sent) if active_side == 'right' else (sent, held_position))


def main(check_only, *, checkpoint, model, task='bowls', duration_s=3.0, record_rollout=True,
         right_only=False, left_only=False, right_sleep=False,
         horizon=HORIZON, inference_mode='sync', arm_speed=5.0, gripper_speed=15.0,
         rollout_hz=DEFAULT_ROLLOUT_HZ):
    if not math.isfinite(rollout_hz) or not 0 < rollout_hz <= SERVO_HZ:
        raise ValueError(f'Rollout rate must be positive and at most {SERVO_HZ:g} Hz.')
    action_period = 1 / rollout_hz
    servo_steps = max(1, round(SERVO_HZ / rollout_hz))
    servo_period = action_period / servo_steps
    import base64

    prompt = TRAINING_PROMPTS[task]
    config_path = LEROBOT_ROOT / 'lerobot/configs/robot/aloha.yaml'
    robot_config = init_hydra_config(str(config_path), ['~leader_arms', '~cameras'])
    robot_config.calibration_dir = str(LEROBOT_ROOT / robot_config.calibration_dir)
    robot = make_robot(robot_config)
    if right_sleep and (task != 'headphones' or right_only):
        raise ValueError('Right sleep mode requires the headphones task and left-arm inference dispatch.')
    left_active = left_only or right_sleep
    single_follower = right_only or left_active
    active_side = 'left' if left_active else 'right'
    held_slice = slice(9, 18) if left_active else slice(0, 9)
    if single_follower and (
        tuple(robot.follower_arms) != ('left', 'right')
        or any(len(bus.motor_names) != 9 for bus in robot.follower_arms.values())
    ):
        raise RuntimeError('Single-follower dispatch requires nine motors per follower in left, right order.')
    motion = AlohaInferenceMotion(robot, LEROBOT_ROOT / '.cache/poses/ceremony_pose.json', 5, 15, {})
    if right_sleep:
        pose = json.loads((LEROBOT_ROOT / '.cache/poses/rest_pose.json').read_text())
        saved = pose['joint_values']['right_follower']
        names = robot.follower_arms['right'].motor_names
        positions = dict(zip(saved['motor_names'], saved['positions'], strict=True))
        if pose.get('robot_type') != 'aloha' or len(saved['motor_names']) != len(names) or positions.keys() != set(names):
            raise ValueError('Sleep pose motor names must match the ALOHA right follower.')
        startup_target = motion.ceremony_target.clone()
        startup_target[held_slice] = torch.tensor([positions[name] for name in names])
        motion.ceremony_target = motion._target(startup_target)
    motion.speed[:] = 5
    motion.speed[[8, 17]] = 15
    configs = json.loads(RIG.read_text())['arms']
    calibrations = {
        side: (json.loads((robot.calibration_dir / f'{side}_follower.json').read_text()),
               json.loads(Path(configs[side]['calibration']).read_text()))
        for side in robot.follower_arms
    }
    pipelines = {}
    camera_cleanup = ExitStack()
    frames_seen = {}

    def check_health():
        for side, bus in robot.follower_arms.items():
            follower_power(side, bus)

    def observation_state():
        state = []
        for side, bus in robot.follower_arms.items():
            aloha, training = calibrations[side]
            names = bus.motor_names
            bus.set_calibration(None)
            try:
                ticks = bus.read('Present_Position')
                bus.set_calibration(training)
                values = bus.apply_calibration(ticks.copy(), names)
            finally:
                bus.set_calibration(aloha)
            cfg = configs[side]
            dofs = np.array([values[names.index(joint)] for joint in JOINTS])
            state.extend(np.deg2rad((dofs - cfg['offset_deg']) / cfg['sign']).tolist())
            state.append(float(np.clip(values[names.index('gripper')] / 100, 0, 1)))
        return state

    stop_observations = threading.Event()

    def collect_frames(deadline, rgb_timeout=None):
        frames = {}
        rgb_deadline = time.monotonic() + rgb_timeout if rgb_timeout is not None else None
        while len(frames) != len(SERIALS):
            now = time.monotonic()
            if rgb_deadline is not None and rgb_deadline <= deadline and now >= rgb_deadline:
                missing = ', '.join(f'{role} ({serial})' for role, serial in SERIALS.items() if role not in frames)
                raise CameraStallError(f'No fresh RGB frame for {rgb_timeout:g}s from {missing}.')
            if stop_observations.is_set() or now >= deadline:
                missing = ', '.join(f'{role} ({serial})' for role, serial in SERIALS.items() if role not in frames)
                raise TimeoutError(f'Waiting for fresh RGB images from {missing}')
            for role, pipeline in pipelines.items():
                if role in frames:
                    continue
                if recording is not None:
                    recording.check()
                    latest = recording.latest.get(role)
                    if latest and latest[0] != frames_seen.get(role):
                        frames_seen[role], frames[role] = latest
                    continue
                frame = pipeline.poll_for_frames()
                color = frame.get_color_frame() if frame else None
                if color and color.get_frame_number() != frames_seen.get(role):
                    frames[role] = np.asanyarray(color.get_data()).copy()
                    frames_seen[role] = color.get_frame_number()
            if len(frames) != len(SERIALS):
                time.sleep(.002)
        return frames

    def observe(deadline, state=None, *, rgb_timeout=None):
        frames = collect_frames(deadline, rgb_timeout)
        if state is None:
            state = observation_state()
        images = {}
        for role, frame in frames.items():
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, 'JPEG', quality=85)
            images[role] = base64.b64encode(buf.getvalue()).decode('ascii')
        if time.monotonic() >= deadline:
            raise TimeoutError('Observation deadline expired')
        return build_payload(**images, state=state, prompt=prompt)

    def convert(action):
        action = np.asarray(action, dtype=float)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('WAMU action must contain 14 finite joint and gripper values')
        targets = []
        for (side, bus), q in zip(robot.follower_arms.items(), action.reshape(2, 7), strict=True):
            if left_active and side == 'right':
                targets.extend(motion.ceremony_target[held_slice].tolist())
                continue
            aloha, training = calibrations[side]
            cfg = configs[side]
            pos = dict(zip(JOINTS, np.rad2deg(q[:6]) * cfg['sign'] + cfg['offset_deg'], strict=True))
            pos.update(shoulder_shadow=pos['shoulder'], elbow_shadow=pos['elbow'], gripper=q[6] * 100)
            names = bus.motor_names
            bus.set_calibration(training)
            try:
                ticks = bus.revert_calibration(np.array([pos[name] for name in names], dtype=float), names)
            finally:
                bus.set_calibration(aloha)
            if abs(int(ticks[1]) - int(ticks[2])) > 2 or abs(int(ticks[3]) - int(ticks[4])) > 2:
                raise ValueError(f'{side}: incompatible primary and shadow targets')
            targets.extend(bus.apply_calibration(ticks.copy(), names).tolist())
        target = torch.tensor(targets, dtype=torch.float32)
        # Quantization near gripper endpoints can drift a fraction of a percent.
        target[[8, 17]] = target[[8, 17]].clamp(0, 100)
        return motion._target(target)

    def request(ws, payload, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError('WAMU deadline expired before request')
        ws.send(json.dumps(dict(payload, type='obs')))
        raw = ws.recv(timeout=max(0, deadline - time.monotonic()))
        response = json.loads(raw)
        if response.get('type') != 'action':
            raise RuntimeError(f'WAMU response: {response.get("type")}')
        return response['action']

    def reset(ws, timeout):
        ws.send('{"type":"reset"}')
        if json.loads(ws.recv(timeout=timeout)).get('type') != 'reset_ack':
            raise RuntimeError('WAMU reset was not acknowledged')

    with (OUT / ('timed-test-preflight.jsonl' if check_only else 'horizon8-hardware.jsonl')).open('a', buffering=1) as log:
        recording = None
        if record_rollout and not check_only:
            episode_root = REAL_RESULTS / task / model / checkpoint.name
            recording = RolloutRecording(episode_root, task)
        def record(event, **details):
            line = json.dumps(dict(time=time.monotonic(), event=event, **details)) + '\n'
            log.write(line)
            if recording is not None:
                recording.events.write(line)

        motors = [(side, name) for side, bus in robot.follower_arms.items() for name in bus.motor_names]
        record('configuration', task=task, model=model, prompt=prompt, duration_s=duration_s, fps=rollout_hz, servo_hz=1 / servo_period, horizon=horizon,
               inference_mode=inference_mode, arm_speed_deg_s=arm_speed, gripper_speed_pct_s=gripper_speed,
               right_only=right_only, left_only=left_only, right_sleep=right_sleep,
               checkpoint=str(checkpoint), motor_order=[f"{side}.{name}" for side, name in motors])
        print(f'Task {task}, closed-loop duration {duration_s:g}s, {rollout_hz:g} Hz, horizon {horizon}, '
              f'{inference_mode} inference, rollout speed caps {arm_speed:g}°/s arm and '
              f'{gripper_speed:g}%/s gripper (0 = uncapped).', flush=True)

        read = motion._positions
        send = robot.send_action
        held_position = None
        last_health = 0
        run_deadline = None
        hold_required = False
        stop_signal = None
        handlers = {}
        begun = None
        paused_at = None
        paused_seconds = 0.0
        count = 0

        def on_stop(signum, frame):
            nonlocal stop_signal
            if stop_signal is None:
                stop_signal = signal.Signals(signum).name
                raise KeyboardInterrupt(stop_signal)

        def read_live():
            nonlocal last_health
            now = time.monotonic()
            if now - last_health > .25:
                check_health()
                last_health = now
            actual = read()
            record('feedback', measured=actual.tolist())
            return actual

        def dispatch(target):
            if held_position is None:
                return send(target)
            return send_one_follower(robot, send, target, active_side, held_position)

        def send_live(target):
            if run_deadline is not None and time.monotonic() >= run_deadline:
                raise TimeoutError('Motor-write deadline reached')
            sent = dispatch(target)
            record('command', target=sent.tolist())
            return sent

        gripper_lag = {}
        last_effort_sample = 0.0

        def check_tracking(actual, now):
            for index, (side, name) in enumerate(motors):
                target = motion.last_command[index].item()
                measured = actual[index].item()
                error = target - measured
                if abs(error) <= 5:
                    gripper_lag.pop(index, None)
                    continue
                unit = 'degrees'
                if name == 'gripper':
                    unit = 'percentage points'
                    closing = error < 0
                    previous = gripper_lag.get(index)
                    if previous is None or previous[0] != closing:
                        gripper_lag[index] = (closing, now, False)
                        continue
                    _, since, contact_logged = previous
                    if now - since < 0.30:
                        continue
                    if closing:
                        # In current-based position mode, a loaded, closing jaw
                        # can hold an object before reaching the requested position.
                        bus = robot.follower_arms[side]
                        effort = abs(int(bus.read('Present_Current', name)[0]))
                        limit = abs(int(bus.read('Goal_Current', name)[0]))
                        if limit and effort >= 0.8 * limit:
                            if not contact_logged:
                                record('possible_grasp_contact', motor=f'{side}.{name}',
                                       target=target, measured=measured, current_raw=effort,
                                       goal_current_raw=limit)
                                gripper_lag[index] = (closing, since, True)
                            continue
                age = now - motion.last_write
                record('tracking_fault', motor=f'{side}.{name}', target=target, measured=measured,
                       error=error, unit=unit, seconds_since_write=age)
                raise RuntimeError(f'{side}.{name} tracking error {error:+.3f} {unit}: '
                                   f'command {target:.3f}, measured {measured:.3f}, '
                                   f'{age * 1000:.1f} ms after write.')

        def sample_shoulders(actual):
            for side, bus in robot.follower_arms.items():
                names = ['shoulder', 'shoulder_shadow']
                indices = [motors.index((side, name)) for name in names]
                record('shoulder_effort', side=side, motors=names,
                       command_deg=motion.last_command[indices].tolist(),
                       measured_deg=actual[indices].tolist(),
                       current_raw=bus.read('Present_Current', names).tolist(),
                       pwm_raw=bus.read('Present_PWM', names).tolist(),
                       voltage_v=(bus.read('Present_Input_Voltage', names) * 0.1).tolist(),
                       temperature_c=bus.read('Present_Temperature', names).tolist())

        def send_servo_command(command, dispatch_at):
            nonlocal last_effort_sample
            if motion.cancelled():
                return None
            actual = motion._positions()
            now = time.monotonic()
            check_tracking(actual, now)
            if now - last_effort_sample >= 1.0:
                sample_shoulders(actual)
                last_effort_sample = now
            time.sleep(max(0.0, dispatch_at - time.monotonic()))
            write_at = time.monotonic()
            if motion.cancelled():
                return None
            sent = send_live(command)
            motion.last_write = write_at
            motion.last_command = torch.as_tensor(sent, dtype=torch.float32).clone()
            if motion.last_command.shape != motion.lower.shape or not torch.isfinite(motion.last_command).all():
                raise RuntimeError('Invalid ALOHA command acknowledgement.')
            return sent

        motion._positions = read_live
        robot.send_action = send_live
        operation = 'inference server verification'
        try:
            for sig in (signal.SIGINT, signal.SIGHUP, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, on_stop)
            with connect(f'ws://{SERVER_HOST}:{SERVER_PORT}', open_timeout=2, close_timeout=.2, compression=None,
                         ping_interval=None, max_size=None, proxy=None) as ws:
                verify_checkpoint(ws, checkpoint, horizon, inference_mode)
                operation = 'follower serial startup and initial position read'
                initial = read_startup_positions(robot, calibrations, read, record)
                operation = 'ceremony start pose check'
                arm_mask = torch.arange(18) % 9 != 8
                travel = (initial - motion.ceremony_target).abs()
                max_arm_travel = travel[arm_mask].max().item()
                max_shoulder_elbow_travel = travel[[1, 2, 3, 4, 10, 11, 12, 13]].max().item()
                record('startup_pose', measured=initial.tolist(), target=motion.ceremony_target.tolist(),
                       max_arm_travel_deg=max_arm_travel,
                       max_shoulder_elbow_travel_deg=max_shoulder_elbow_travel)
                if max_arm_travel > 168 or max_shoulder_elbow_travel > 58:
                    raise RuntimeError(
                        f'Start pose exceeds checked ceremony travel. Maximum arm joint travel '
                        f'{max_arm_travel:.1f}° (limit 168°). Maximum shoulder/elbow travel '
                        f'{max_shoulder_elbow_travel:.1f}° (limit 58°).'
                    )
                time.sleep(.3)
                drift = (read() - initial).abs()
                exceeded = drift > motion.settle_drift
                if exceeded.any():
                    motor_index = (drift / motion.settle_drift).argmax().item()
                    side, name = motors[motor_index]
                    unit = 'percentage points' if name == 'gripper' else 'degrees'
                    raise RuntimeError(
                        f'{side}.{name} feedback changed by {drift[motor_index]:.2f} {unit} '
                        f'during the pre-ceremony check (limit {motion.settle_drift[motor_index]:.2f}).'
                    )
                operation = 'camera stream startup'
                for role, serial in SERIALS.items():
                    pipeline = open_camera(role, serial)
                    camera_cleanup.callback(pipeline.stop)
                    pipelines[role] = pipeline
                if recording is not None:
                    recording.start(pipelines)
                # Verify fresh RGB before starting the physical sequence.
                operation = 'fresh RGB and robot observation'
                observe(time.monotonic() + CAMERA_STARTUP_TIMEOUT_S)
                operation = 'ceremony and rollout'
                if not check_only:
                    establish_startup_hold(robot.follower_arms)
                    check_health()
                    hold_required = True
                    record('ceremony_start', measured=initial.tolist(), target=motion.ceremony_target.tolist())
                    print('Starting the opening ceremony with the right follower targeting sleep.'
                          if right_sleep else 'Starting the standard opening ceremony.', flush=True)
                    if not opening_ceremony(motion):
                        raise RuntimeError("Opening ceremony interrupted.")
                    actual = read()
                    record('ceremony_settled', measured=actual.tolist())
                    err = (actual - motion.ceremony_target).abs()[arm_mask].max().item()
                    print(f'Ceremony settled with worst arm-joint error {err:.2f}°.', flush=True)
                    if single_follower:
                        held_position = motion.last_command[held_slice].clone()
                        if right_sleep:
                            if ((actual[held_slice] - held_position).abs() > 5).any():
                                raise RuntimeError('Right follower exceeded 5 degrees / percentage points from sleep.')
                            print('Left-only inference dispatch. Right follower keeps its sleep goal.', flush=True)
                        elif left_only:
                            if ((actual[held_slice] - motion.ceremony_target[held_slice]).abs() > 5).any():
                                raise RuntimeError('Right follower exceeded 5 degrees / percentage points from ceremony.')
                            print('Left-only inference dispatch. Right follower keeps its ceremony goal.', flush=True)
                        else:
                            print('Right-only inference dispatch. Left follower keeps its ceremony goal.', flush=True)
                reset(ws, 1)
                observation = observe(time.monotonic() + 2)
                prediction_start = time.monotonic()
                try:
                    target = convert(request(ws, observation, prediction_start + 15))
                except TimeoutError as exc:
                    raise TimeoutError('WAMU first prediction exceeded its 15-second startup timeout.') from exc
                record('first_prediction_latency', seconds=time.monotonic() - prediction_start)
                start = read()
                arm_mask = torch.arange(18) % 9 != 8
                delta = (target - start).abs()
                print(f'First WAMU pose: max arm changes {delta[:8].max():.2f}° left, '
                      f'{delta[9:17].max():.2f}° right; grippers {target[8]:.1f}%, {target[17]:.1f}%.', flush=True)
                record('first_prediction', target=target.tolist(), measured=start.tolist())
                if single_follower:
                    target[held_slice] = start[held_slice] if check_only else held_position
                    arm_mask[held_slice] = False
                if (delta[arm_mask] > 10).any():
                    raise RuntimeError('First predicted arm move exceeds 10° from ceremony.')
                if check_only:
                    return
                if not motion._move(target):
                    raise RuntimeError('First pose was interrupted')
                record('first_pose_settled', measured=read().tolist())
                print(f'First pose settled. Starting {duration_s:g}s closed-loop trial.', flush=True)
                reset(ws, 1)
                # Prime the first chunk before starting the timed control clock.
                # Later chunks overlap execution when using the async executor.
                prime_start = time.monotonic()
                goal = convert(request(ws, observe(prime_start + 2), prime_start + 15))
                record('rollout_prime_latency', seconds=time.monotonic() - prime_start)
                begun = time.monotonic()
                run_deadline = begun + duration_s
                count = 0
                reason = 'Timed trial complete'
                # Camera and websocket waits run off the motor thread. All bus reads,
                # calibration changes, action conversion and writes stay on this thread.
                operation = 'closed-loop rollout'
                def fetch_action(state, reset_policy=False):
                    if reset_policy:
                        reset(ws, 1)
                    payload = observe(run_deadline, state, rgb_timeout=ROLLOUT_RGB_TIMEOUT_S)
                    return request(ws, payload, run_deadline)

                pool = ThreadPoolExecutor(max_workers=1)
                pending = None
                knots = [motion.last_command.clone()] * 4
                phase = 0
                try:
                    while time.monotonic() < run_deadline:
                        if phase == 0:
                            if pending is not None and pending.done():
                                if paused_at is not None:
                                    pending.result()  # All three cameras delivered a frame.
                                    resumed = time.monotonic()
                                    pause = resumed - paused_at
                                    paused_seconds += pause
                                    run_deadline = begun + duration_s + paused_seconds
                                    paused_at = None
                                    record('rgb_resumed', pause_seconds=pause)
                                    print(f'RGB restored after {pause:.2f}s. Resuming inference.', flush=True)
                                    # Discard probe images and queued policy actions. Read current
                                    # motor feedback on this thread for the first new observation.
                                    pending = pool.submit(fetch_action, observation_state(), True)
                                else:
                                    try:
                                        goal = convert(pending.result())
                                    except CameraStallError as exc:
                                        paused_at = time.monotonic()
                                        run_deadline = math.inf
                                        record('rgb_hold', message=str(exc))
                                        print(f'{exc} Holding until fresh RGB returns.', flush=True)
                                        pending = pool.submit(collect_frames, math.inf)
                                    else:
                                        pending = None
                            waypoint = knots[-1].clone()
                            if goal is not None:
                                if held_position is not None:
                                    goal[held_slice] = held_position
                                record('prediction', target=goal.tolist())
                                if inference_mode == 'sync' and count % horizon == 0:
                                    record('fresh_chunk', action_index=count,
                                           latency_ms=(time.monotonic() - tick) * 1000 if count else 0)
                                waypoint = goal.clone()
                                if arm_speed > 0:
                                    delta = goal[arm_mask] - knots[-1][arm_mask]
                                    waypoint[arm_mask] = knots[-1][arm_mask] + delta.clamp(
                                        -arm_speed * action_period, arm_speed * action_period)
                                if gripper_speed > 0:
                                    delta = goal[[8, 17]] - motion.last_command[[8, 17]]
                                    waypoint[[8, 17]] = motion.last_command[[8, 17]] + delta.clamp(
                                        -gripper_speed * action_period, gripper_speed * action_period)
                                count += 1
                                goal = None
                                tick = time.monotonic()
                                pending = pool.submit(fetch_action, observation_state())
                            # Repeating the latest knot decelerates the arms to rest
                            # when inference stalls. No old actions are extrapolated.
                            knots = [*knots[1:], waypoint]
                        # Uniform cubic B-spline: position, velocity and acceleration
                        # match at knot boundaries. Convex weights preserve joint bounds.
                        u = (phase + 1) / servo_steps
                        a, b, c, d = knots
                        command = ((1-u)**3 * a + (3*u**3 - 6*u*u + 4) * b
                                   + (-3*u**3 + 3*u*u + 3*u + 1) * c + u**3 * d) / 6
                        command[[8, 17]] = d[[8, 17]]
                        next_write = motion.last_write + servo_period
                        if next_write >= run_deadline or send_servo_command(command, next_write) is None:
                            break
                        phase = (phase + 1) % servo_steps
                except TimeoutError as exc:
                    reason = str(exc) or 'WAMU response deadline reached'
                finally:
                    # Wake a pending receive promptly on cancellation before cleanup.
                    stop_observations.set()
                    ws.close()
                    pool.shutdown(wait=True, cancel_futures=True)
        except KeyboardInterrupt:
            if stop_signal is None:
                raise
        except Exception as exc:
            record('error', phase=operation, error_type=type(exc).__name__, message=str(exc))
            raise
        finally:
            try:
                # Further stop signals must not interrupt the final motor write.
                for sig in handlers:
                    signal.signal(sig, signal.SIG_IGN)
                if hold_required:
                    stopped = time.monotonic()
                    if paused_at is not None:
                        paused_seconds += stopped - paused_at
                    elapsed = stopped - begun - paused_seconds if begun is not None else None
                    run_deadline = None
                    check_health()
                    hold = read()
                    sent = dispatch(hold)
                    time.sleep(.35)
                    final = read()
                    check_health()
                    joints = torch.arange(len(hold)) % 9 != 8
                    drift = (hold - final).abs()[joints].max().item()
                    record('hold', signal=stop_signal, updates=count, control_seconds=elapsed, rgb_pause_seconds=paused_seconds,
                           drift_deg=drift, target=(hold if held_position is None else sent).tolist(),
                           measured=final.tolist())
                    if stop_signal is None and begun is not None and sys.exc_info()[0] is None:
                        print(f'{reason}. {count} updates in {elapsed:.3f}s. Torque on, zero motor faults; '
                              f'max arm drift after hold {drift:.3f}°.', flush=True)
            finally:
                try:
                    try:
                        if recording is not None:
                            record('end', signal=stop_signal)
                            recording.close()
                    finally:
                        camera_cleanup.close()
                finally:
                    try:
                        for bus in robot.follower_arms.values():
                            if bus.is_connected:
                                bus.disconnect()
                        robot.is_connected = False
                    finally:
                        for sig, previous in handlers.items():
                            signal.signal(sig, previous)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TRAINING_PROMPTS, default='bowls',
                        help='Select the exact training prompt (default: bowls).')
    parser.add_argument('--rollout-hz', type=float, default=DEFAULT_ROLLOUT_HZ,
                        help='Policy action rate in Hz (default: 15). Lower rates allow more time per action.')
    parser.add_argument('--duration-s', '--duration', type=positive_seconds, default=3.0,
                        help='Closed-loop duration in seconds after startup (default: 3).')
    parser.add_argument('--model', choices=MODELS, default=MODELS[0], help='WAMU checkpoint family.')
    parser.add_argument('--weights', '--checkpoint-step', type=int, choices=WEIGHT_STEPS, default=20000,
                        help='WAMU checkpoint step (default: 20000).')
    parser.add_argument('--record', action=argparse.BooleanOptionalAction, default=True,
                        help='Save hardware episodes under real_results (enabled by default).')
    parser.add_argument('--training-control', action='store_true',
                        help='Use 32-action chunks, async inference, and no rollout speed cap.')
    parser.add_argument('--horizon', type=int, choices=range(1, 33), default=None,
                        help='Actions per inference horizon (default: 8, or 32 with --training-control).')
    parser.add_argument('--inference-mode', choices=('sync', 'async'), default=None,
                        help='Chunk scheduling (default: sync, or async with --training-control).')
    parser.add_argument('--arm-speed-deg-s', type=nonnegative_speed, default=None,
                        help='Rollout arm speed cap, 0 disables (default: 5, or 0 with --training-control).')
    parser.add_argument('--gripper-speed-pct-s', type=nonnegative_speed, default=None,
                        help='Rollout gripper speed cap, 0 disables (default: 15, or 0 with --training-control).')
    followers = parser.add_mutually_exclusive_group()
    followers.add_argument('--right-only', action='store_true',
                        help='After the standard ceremony, dispatch inference and final hold to the right follower only.')
    followers.add_argument('--left-only', action='store_true',
                           help='After the standard ceremony, dispatch inference and final hold to the left follower only.')
    followers.add_argument('--right-sleep', action='store_true',
                           help='For headphones, start the right follower at sleep and dispatch inference only to the left.')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only', action='store_true', help='Read observations and predict without moving.')
    modes.add_argument('--serve-only', action='store_true', help='Start the selected inference server without opening motor buses.')
    modes.add_argument('--client-only', action='store_true', help='Run the trial using an already running inference server.')
    modes.add_argument('--auto-serve', action='store_true', help='Manage the local inference server for this run.')
    args = parser.parse_args()
    if args.right_sleep and args.task != 'headphones':
        parser.error('--right-sleep requires --task headphones')
    horizon = args.horizon if args.horizon is not None else (32 if args.training_control else HORIZON)
    inference_mode = args.inference_mode or ('async' if args.training_control else 'sync')
    arm_speed = args.arm_speed_deg_s if args.arm_speed_deg_s is not None else (0.0 if args.training_control else 5.0)
    gripper_speed = args.gripper_speed_pct_s if args.gripper_speed_pct_s is not None else (0.0 if args.training_control else 15.0)
    if not 0 < args.rollout_hz <= SERVO_HZ:
        parser.error(f'--rollout-hz must be positive and at most {SERVO_HZ:g}')
    control = dict(rollout_hz=args.rollout_hz, horizon=horizon, inference_mode=inference_mode, arm_speed=arm_speed, gripper_speed=gripper_speed)
    try:
        checkpoint = find_checkpoint(args.model, args.weights)
    except ValueError as exc:
        parser.error(str(exc))
    if args.serve_only:
        serve_checkpoint(checkpoint, horizon, inference_mode)
    elif args.auto_serve:
        with selected_server(checkpoint, args.model, args.weights, horizon, inference_mode):
            main(False, checkpoint=checkpoint, model=args.model, task=args.task, duration_s=args.duration_s,
                 record_rollout=args.record, right_only=args.right_only, left_only=args.left_only,
                 right_sleep=args.right_sleep, **control)
    else:
        main(args.check_only, checkpoint=checkpoint, model=args.model, task=args.task, duration_s=args.duration_s,
             record_rollout=args.record, right_only=args.right_only, left_only=args.left_only,
             right_sleep=args.right_sleep, **control)
