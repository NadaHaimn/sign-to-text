import atexit
import os
import json
import urllib.request
import threading
import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except Exception:  # pragma: no cover - handled at runtime with a clear error
    mp = None
    mp_python = None
    mp_vision = None

HAND_TASK_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_TASK_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

_HAND_LANDMARKER = None
_POSE_LANDMARKER = None
_LANDMARKER_LOCK = threading.Lock()
_LANDMARKER_CLEANUP_REGISTERED = False

def load_label_map(data_dir=None):
    if data_dir is None:
        base_dir = os.path.dirname(__file__)
        data_dir = os.path.join(base_dir, "data")
        search_dirs = [data_dir, base_dir]
    else:
        search_dirs = [data_dir]

    candidates = ["label_map_arabic.json", "label_map.json"]
    for search_dir in search_dirs:
        for c in candidates:
            p = os.path.join(search_dir, c)
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
    return None

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
        # fallback to reading current frame even if failed earlier
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            return frame
        raise RuntimeError("Could not read frame from video")
    finally:
        cap.release()

def preprocess_frame(frame, target_shape):
    # target_shape is typically (None, H, W, C) or (None, C, H, W)
    if frame is None:
        raise ValueError("frame is None")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # infer H, W
    if len(target_shape) == 4:
        _, h, w, c = target_shape
        if c not in (1, 3):
            # maybe channels-first
            _, c2, h2, w2 = target_shape
            h, w = h2, w2
        if h is None or w is None:
            h, w = frame.shape[0], frame.shape[1]
    else:
        h, w = frame.shape[0], frame.shape[1]
    h, w = int(h), int(w)
    resized = cv2.resize(frame, (w, h))
    arr = resized.astype("float32") / 255.0
    return arr


def resample_sequence(sequence, target_len=48):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2:
        raise ValueError("sequence must have shape (frames, features)")
    if len(sequence) == 0:
        raise ValueError("sequence is empty")
    if len(sequence) == target_len:
        return sequence
    if len(sequence) == 1:
        return np.repeat(sequence, target_len, axis=0)

    old_idx = np.linspace(0.0, 1.0, num=len(sequence), dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    out = np.zeros((target_len, sequence.shape[1]), dtype=np.float32)
    for feature_idx in range(sequence.shape[1]):
        out[:, feature_idx] = np.interp(new_idx, old_idx, sequence[:, feature_idx])
    return out


def _landmarks_to_array(landmarks, count):
    arr = np.zeros((count, 3), dtype=np.float32)
    if landmarks is None:
        return arr

    limit = min(count, len(landmarks))
    for idx in range(limit):
        lm = landmarks[idx]
        arr[idx] = (lm.x, lm.y, lm.z)
    return arr


def _ensure_task_file(url, path):
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(url, path)
    return path


def _close_landmarkers():
    global _HAND_LANDMARKER, _POSE_LANDMARKER

    for landmarker in (_HAND_LANDMARKER, _POSE_LANDMARKER):
        if landmarker is not None:
            close = getattr(landmarker, "close", None)
            if callable(close):
                close()

    _HAND_LANDMARKER = None
    _POSE_LANDMARKER = None


def _get_landmarkers():
    global _HAND_LANDMARKER, _POSE_LANDMARKER, _LANDMARKER_CLEANUP_REGISTERED

    if _HAND_LANDMARKER is not None and _POSE_LANDMARKER is not None:
        return _HAND_LANDMARKER, _POSE_LANDMARKER

    with _LANDMARKER_LOCK:
        if _HAND_LANDMARKER is not None and _POSE_LANDMARKER is not None:
            return _HAND_LANDMARKER, _POSE_LANDMARKER

        base_dir = os.path.join(os.path.dirname(__file__), "data", "mediapipe_tasks")
        hand_task_path = _ensure_task_file(HAND_TASK_URL, os.path.join(base_dir, "hand_landmarker.task"))
        pose_task_path = _ensure_task_file(POSE_TASK_URL, os.path.join(base_dir, "pose_landmarker_lite.task"))

        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_task_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
        )
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_task_path),
            running_mode=mp_vision.RunningMode.IMAGE,
        )

        _HAND_LANDMARKER = mp_vision.HandLandmarker.create_from_options(hand_options)
        _POSE_LANDMARKER = mp_vision.PoseLandmarker.create_from_options(pose_options)

        if not _LANDMARKER_CLEANUP_REGISTERED:
            atexit.register(_close_landmarkers)
            _LANDMARKER_CLEANUP_REGISTERED = True

        return _HAND_LANDMARKER, _POSE_LANDMARKER


def preload_landmarker_assets():
    if mp is None or mp_python is None or mp_vision is None:
        raise RuntimeError("mediapipe is required for landmark extraction")

    _get_landmarkers()


def normalize_single_frame(pose_xyz, left_hand_xyz, right_hand_xyz):
    pose = np.asarray(pose_xyz, dtype=np.float32).reshape(-1, 3).copy()
    left_hand = np.asarray(left_hand_xyz, dtype=np.float32).reshape(-1, 3).copy()
    right_hand = np.asarray(right_hand_xyz, dtype=np.float32).reshape(-1, 3).copy()

    left_shoulder = pose[11]
    right_shoulder = pose[12]
    shoulders_valid = np.any(left_shoulder != 0.0) and np.any(right_shoulder != 0.0)

    if shoulders_valid:
        center = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)
    else:
        center = pose[0]
        scale = 1.0

    scale = max(float(scale), 1e-6)
    pose = (pose - center) / scale
    left_hand = (left_hand - center) / scale
    right_hand = (right_hand - center) / scale

    return np.concatenate([pose.reshape(-1), left_hand.reshape(-1), right_hand.reshape(-1)], axis=0).astype(np.float32)


def extract_landmark_sequence(video_path, max_frames=12, frame_stride=8, max_side=320):
    if mp is None or mp_python is None or mp_vision is None:
        raise RuntimeError("mediapipe is required for landmark extraction")

    frames = []
    cap = cv2.VideoCapture(video_path)
    try:
        hand_landmarker, pose_landmarker = _get_landmarkers()
        frame_count = 0
        frame_index = 0
        frame_stride = max(1, int(frame_stride))

        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_index % frame_stride != 0:
                frame_index += 1
                continue

            if max_side is not None:
                height, width = frame.shape[:2]
                longest_side = max(height, width)
                if longest_side > max_side:
                    scale = max_side / float(longest_side)
                    resized_width = max(1, int(width * scale))
                    resized_height = max(1, int(height * scale))
                    frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_landmarker.detect(image)
            pose_result = pose_landmarker.detect(image)

            pose_xyz = _landmarks_to_array(pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None, 33)
            left_hand_xyz = np.zeros((21, 3), dtype=np.float32)
            right_hand_xyz = np.zeros((21, 3), dtype=np.float32)

            if hand_result.hand_landmarks:
                for idx, hand_lms in enumerate(hand_result.hand_landmarks):
                    handedness_name = None
                    if hand_result.handedness and idx < len(hand_result.handedness) and len(hand_result.handedness[idx]) > 0:
                        handedness_name = hand_result.handedness[idx][0].category_name.lower()

                    coords = _landmarks_to_array(hand_lms, 21)
                    if handedness_name == "left":
                        left_hand_xyz = coords
                    elif handedness_name == "right":
                        right_hand_xyz = coords

            frames.append(normalize_single_frame(pose_xyz, left_hand_xyz, right_hand_xyz))
            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

            frame_index += 1
    finally:
        cap.release()

    if not frames:
        frames.append(np.zeros(225, dtype=np.float32))

    return resample_sequence(np.stack(frames).astype(np.float32), target_len=48)
