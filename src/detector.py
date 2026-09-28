from pathlib import Path
import time

from ultralytics import YOLO

from src.config import settings


# Actual classes in this road-damage dataset
CLASS_NAMES = {
    0: "Pothole",
    1: "Crack",
    2: "Manhole",
}


class RoadDamageDetector:

    def __init__(self, model_path: Path):

        self.model_path = Path(model_path)

        self.model = (
            YOLO(str(self.model_path))
            if self.model_path.exists()
            else None
        )

    def is_ready(self):
        return self.model is not None

    def predict(self, image_path: Path, save_dir: Path):

        if not self.model:
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        start = time.perf_counter()

        results = self.model.predict(
            source=str(image_path),
            conf=settings.confidence,
            save=True,
            project=str(save_dir.parent),
            name=save_dir.name,
            exist_ok=True,
            verbose=False,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        result = results[0]

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                bbox = [
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ]

                damage_name = CLASS_NAMES.get(
                    class_id,
                    f"Unknown Class {class_id}"
                )

                detections.append(
                    {
                        "damage_type": damage_name,
                        "class_id": class_id,
                        "confidence": round(
                            confidence,
                            4
                        ),
                        "bbox": bbox,
                    }
                )

        image_candidates = list(
            save_dir.glob("*")
        )

        annotated = next(
            (
                p
                for p in image_candidates
                if p.suffix.lower()
                in {".jpg", ".jpeg", ".png"}
            ),
            None,
        )

        return {
            "detections": detections,
            "annotated_path": annotated,
            "latency_ms": round(
                latency_ms,
                2
            ),
        }