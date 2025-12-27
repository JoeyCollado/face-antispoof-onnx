import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_MODEL = MODELS_DIR / "detector.onnx"
LIVENESS_MODEL = MODELS_DIR / "best_model_quantized.onnx"


def preprocess(img: np.ndarray, model_img_size: int) -> np.ndarray:
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
    if not face_crops:
        raise ValueError("face_crops list cannot be empty")

    batch = np.zeros(
        (len(face_crops), 3, model_img_size, model_img_size), dtype=np.float32
    )
    for i, face_crop in enumerate(face_crops):
        batch[i] = preprocess(face_crop, model_img_size)

    return batch


def crop(img: np.ndarray, bbox: tuple, bbox_expansion_factor: float) -> np.ndarray:
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


def process_with_logits(raw_logits: np.ndarray, threshold: float) -> Dict:
    real_logit = float(raw_logits[0])
    spoof_logit = float(raw_logits[1])
    logit_diff = real_logit - spoof_logit
    is_real = logit_diff >= threshold
    confidence = abs(logit_diff)

    return {
        "is_real": bool(is_real),
        "status": "real" if is_real else "spoof",
        "logit_diff": float(logit_diff),
        "real_logit": float(real_logit),
        "spoof_logit": float(spoof_logit),
        "confidence": float(confidence),
    }


def infer(
    face_crops: List[np.ndarray],
    ort_session: ort.InferenceSession,
    input_name: str,
    model_img_size: int,
) -> List[np.ndarray]:
    if not face_crops or ort_session is None:
        return []

    try:
        batch_input = preprocess_batch(face_crops, model_img_size)
        logits = ort_session.run([], {input_name: batch_input})[0]

        if logits.shape != (len(face_crops), 2):
            raise ValueError("Model output shape mismatch")

        return [logits[i] for i in range(len(face_crops))]
    except Exception:
        return []


def load_model(model_path: str) -> Tuple[Optional[ort.InferenceSession], Optional[str]]:
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


def load_detector(
    model_path: str,
    input_size: tuple,
    confidence_threshold: float = 0.8,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
):
    if not Path(model_path).exists():
        return None

    try:
        return cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            input_size,
            confidence_threshold,
            nms_threshold,
            top_k,
        )
    except Exception:
        return None


def detect(
    image: np.ndarray, detector, min_face_size: int = 60, margin: int = 5
) -> List[Dict]:
    if detector is None or image is None:
        return []

    img_h, img_w = image.shape[:2]
    detector.setInputSize((img_w, img_h))
    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return []

    detections = []
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        conf = float(face[14])

        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            continue

        dist_left = x
        dist_right = img_w - (x + w)
        dist_top = y
        dist_bottom = img_h - (y + h)
        if min(dist_left, dist_right, dist_top, dist_bottom) < margin:
            continue

        if w >= min_face_size and h >= min_face_size:
            detections.append(
                {
                    "bbox": {
                        "x": float(x),
                        "y": float(y),
                        "width": float(w),
                        "height": float(h),
                    },
                    "confidence": conf,
                }
            )

    return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--model_img_size", type=int, default=128)
    parser.add_argument("--bbox_expansion_factor", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--margin", type=int, default=5)
    parser.add_argument("--detector_model", type=str, default=str(DETECTOR_MODEL))
    parser.add_argument("--liveness_model", type=str, default=str(LIVENESS_MODEL))

    args = parser.parse_args()

    p = max(1e-6, min(1 - 1e-6, args.threshold))
    logit_threshold = np.log(p / (1 - p))

    face_detector = load_detector(args.detector_model, (320, 320))
    liveness_session, input_name = load_model(args.liveness_model)

    if liveness_session is None or face_detector is None:
        exit(1)

    if args.image is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            exit(1)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detect(frame_rgb, face_detector, margin=args.margin)

            if faces:
                face_crops = []
                valid_faces = []
                for face in faces:
                    bbox = face["bbox"]
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    try:
                        face_crop = crop(frame_rgb, (x, y, x + w, y + h), args.bbox_expansion_factor)
                        face_crops.append(face_crop)
                        valid_faces.append((face, (int(x), int(y), int(w), int(h))))
                    except Exception:
                        continue

                if face_crops:
                    predictions = infer(
                        face_crops, liveness_session, input_name, args.model_img_size
                    )

                    for (face, (x, y, w, h)), pred in zip(valid_faces, predictions):
                        try:
                            result = process_with_logits(pred, logit_threshold)
                        except Exception:
                            continue

                        color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                        label = (
                            f"{result['status'].upper()}: {result['logit_diff']:.2f}"
                        )
                        cv2.putText(
                            frame,
                            label,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

            cv2.imshow("Liveness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(args.image)
        if image is None:
            exit(1)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = detect(image_rgb, face_detector, margin=args.margin)

        if not faces:
            exit(0)

        face_crops = []
        valid_faces = []
        for face in faces:
            bbox = face["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            try:
                face_crop = crop(image_rgb, (x, y, x + w, y + h), args.bbox_expansion_factor)
                face_crops.append(face_crop)
                valid_faces.append((int(x), int(y), int(w), int(h)))
            except Exception:
                continue

        if not face_crops:
            exit(0)

        predictions = infer(
            face_crops, liveness_session, input_name, args.model_img_size
        )

        for (x, y, w, h), pred in zip(valid_faces, predictions):
            try:
                result = process_with_logits(pred, logit_threshold)
            except Exception:
                continue

            color = (0, 255, 0) if result["is_real"] else (0, 0, 255)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            label = f"{result['status'].upper()}: {result['logit_diff']:.2f}"
            cv2.putText(
                image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
