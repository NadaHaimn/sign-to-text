import os
import glob
import numpy as np
from tensorflow.keras.models import load_model

from utils import load_label_map, extract_landmark_sequence

_MODEL = None
_LABEL_MAP = None

def find_model_path():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    search_dirs = [data_dir, base_dir]
    candidates = []
    for search_dir in search_dirs:
        candidates.extend(glob.glob(os.path.join(search_dir, "*.keras")))
        candidates.extend(glob.glob(os.path.join(search_dir, "*.h5")))
    if not candidates:
        raise FileNotFoundError("No .keras or .h5 model found in data/ or project root")

    preferred = [
        os.path.join(data_dir, "best_arabic_sign_model.keras"),
        os.path.join(base_dir, "best_arabic_sign_model.keras"),
        os.path.join(data_dir, "final_arabic_sign_model.keras"),
        os.path.join(base_dir, "final_arabic_sign_model.keras"),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path

    return sorted(candidates)[0]

def get_label_map():
    global _LABEL_MAP
    if _LABEL_MAP is None:
        _LABEL_MAP = load_label_map()
    return _LABEL_MAP

def get_model():
    global _MODEL
    if _MODEL is None:
        model_path = find_model_path()
        _MODEL = load_model(model_path, compile=False)
    return _MODEL


def _build_model_inputs(model, sequence_batch):
    if len(getattr(model, "inputs", [])) != 1:
        return sequence_batch

    input_tensor = model.inputs[0]
    input_name = getattr(input_tensor, "name", "")
    if not input_name:
        return sequence_batch

    input_key = input_name.split(":", 1)[0].split("/", 1)[-1]
    return {input_key: sequence_batch}

def predict_video(video_path):
    model = get_model()
    sequence = extract_landmark_sequence(video_path, max_frames=12, frame_stride=8, max_side=320)
    expected_shape = tuple(model.input_shape[1:])
    if sequence.shape != expected_shape:
        raise ValueError(f"Expected input shape {expected_shape}, got {sequence.shape}")

    x = np.expand_dims(sequence, 0)
    model_inputs = _build_model_inputs(model, x)
    preds = model(model_inputs, training=False).numpy()
    if preds.ndim == 2:
        idx = int(preds[0].argmax())
        score = float(preds[0, idx])
    else:
        # fallback
        idx = int(np.argmax(preds))
        score = float(np.max(preds))
    labels = get_label_map()
    label = None
    if labels:
        # labels might be dict index->name or list
        if isinstance(labels, dict):
            label = labels.get(str(idx)) or labels.get(idx)
        elif isinstance(labels, list) and idx < len(labels):
            label = labels[idx]
    return {"label": label or str(idx), "score": score}

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/video.mp4")
        sys.exit(1)
    print(predict_video(sys.argv[1]))
