from models.FaceAntiSpoofing import FaceAntiSpoofingInterface

import onnxruntime as ort
from typing import Tuple, Optional
from pathlib import Path


def load_model(model_path: str) -> Tuple[Optional[ort.InferenceSession], Optional[str]]:
    """Load ONNX model. Return (session, input_name) or (None, None) on failure."""
    if not Path(model_path).exists():
        return None, None

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        available_providers = ort.get_available_providers()
        preferred_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in preferred_providers if p in available_providers]

        if not providers:
            providers = available_providers

        ort_session = ort.InferenceSession(
            model_path, sess_options=sess_options, providers=providers
        )
        input_name = ort_session.get_inputs()[0].name
        return ort_session, input_name
    except Exception:
        return None, None


import cv2
import numpy as np
from typing import List

import numpy as np
import onnxruntime as ort
import sys
from typing import List, Dict


def process_with_logits(raw_logits: np.ndarray, threshold: float) -> Dict:
    """Convert raw logits to real/spoof classification with softmax."""
    # Apply softmax to convert logits to probabilities
    exp_logits = np.exp(raw_logits - np.max(raw_logits))  # for numerical stability
    probabilities = exp_logits / np.sum(exp_logits)

    real_prob = float(probabilities[0])
    spoof_prob = float(probabilities[1])

    # Use probability difference instead of logit difference
    prob_diff = real_prob - spoof_prob
    is_real = prob_diff >= threshold

    # Calculate confidence based on probability (0-1 range)
    confidence = abs(prob_diff)

    return {
        "real_probability": real_prob,
        "spoof_probability": spoof_prob,
        "probability_difference": prob_diff,
        "is_real": bool(is_real),
        "confidence": float(confidence)
    }

def infer(
    face_crops: List[np.ndarray],
    ort_session: ort.InferenceSession,
    input_name: str,
    model_img_size: int,
) -> List[np.ndarray]:
    """Run batch inference on cropped face images. Return list of logits per face."""
    if not face_crops or ort_session is None:
        return []

    try:
        batch_input = preprocess_batch(face_crops, model_img_size)
        logits = ort_session.run([], {input_name: batch_input})[0]

        if logits.shape != (len(face_crops), 2):
            raise ValueError("Model output shape mismatch")

        return [logits[i] for i in range(len(face_crops))]
    except Exception as e:
        print(f"Inference error: {e}", file=sys.stderr)
        return []
def preprocess(img: np.ndarray, model_img_size: int) -> np.ndarray:
    """Resize with letterboxing, normalize to [0,1], convert to CHW."""
    new_size = model_img_size
    old_size = img.shape[:2]

    ratio = float(new_size) / max(old_size)
    scaled_shape = tuple([int(x * ratio) for x in old_size])

    interpolation = cv2.INTER_LANCZOS4 if ratio > 1.0 else cv2.INTER_AREA
    img = cv2.resize(
        img, (scaled_shape[1], scaled_shape[0]), interpolation=interpolation
    )

    delta_w = new_size - scaled_shape[1]
    delta_h = new_size - scaled_shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101)

    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    return img


def preprocess_batch(face_crops: List[np.ndarray], model_img_size: int) -> np.ndarray:
    """Preprocess multiple face crops into a batched array."""
    if not face_crops:
        raise ValueError("face_crops list cannot be empty")

    batch = np.zeros(
        (len(face_crops), 3, model_img_size, model_img_size), dtype=np.float32
    )
    for i, face_crop in enumerate(face_crops):
        batch[i] = preprocess(face_crop, model_img_size)

    return batch


def crop(img: np.ndarray, bbox: tuple, bbox_expansion_factor: float) -> np.ndarray:
    """Extract square face crop from bbox with expansion. Pad edges with reflection."""
    original_height, original_width = img.shape[:2]
    x, y, w, h = bbox

    w = w - x
    h = h - y

    if w <= 0 or h <= 0:
        raise ValueError("Invalid bbox dimensions")

    max_dim = max(w, h)
    center_x = x + w / 2
    center_y = y + h / 2

    x = int(center_x - max_dim * bbox_expansion_factor / 2)
    y = int(center_y - max_dim * bbox_expansion_factor / 2)
    crop_size = int(max_dim * bbox_expansion_factor)

    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(original_width, x + crop_size)
    crop_y2 = min(original_height, y + crop_size)

    top_pad = int(max(0, -y))
    left_pad = int(max(0, -x))
    bottom_pad = int(max(0, (y + crop_size) - original_height))
    right_pad = int(max(0, (x + crop_size) - original_width))

    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    else:
        img = np.zeros((0, 0, 3), dtype=img.dtype)

    result = cv2.copyMakeBorder(
        img,
        top_pad,
        bottom_pad,
        left_pad,
        right_pad,
        cv2.BORDER_REFLECT_101,
    )

    if result.shape[0] != crop_size or result.shape[1] != crop_size:
        raise ValueError(
            f"Crop size mismatch: expected {crop_size}x{crop_size}, got {result.shape[0]}x{result.shape[1]}"
        )

    return result


class M9FaceAntiSpoofing(FaceAntiSpoofingInterface):
    def __init__(self, load_onnx = False):
        self.liveness_session, self.input_name = load_model("models/m9/files/best.onnx")
        threshold = 0.5
        p = max(1e-6, min(1 - 1e-6, threshold))
        self.logit_threshold = np.log(p / (1 - p))
        self.model_size = 128

    def get_real_score(self, bgr, bbox, is_crop = False):
        bbox_expansion_factor = 1.5
        x, y, w, h = bbox
        image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        face_crops = [crop(
            image_rgb, (x, y, x + w, y + h), bbox_expansion_factor
        )]

        pred = infer(face_crops, self.liveness_session, self.input_name, self.model_size)[0]

        result = process_with_logits(pred, self.logit_threshold)

        if result["is_real"]:
            return result['confidence']
        else:
            return 1 - result['confidence']
