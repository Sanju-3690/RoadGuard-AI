def assess_visual_severity(detections, image_size):
    width, height = image_size
    image_area = max(width * height, 1)
    if not detections:
        return {
            "level": "No damage detected", "score": 0.0,
            "indicators": {"damage_count": 0, "largest_bbox_area_ratio": 0.0, "mean_confidence": 0.0},
            "explanation": "No supported road-damage detection exceeded the configured confidence threshold.",
        }
    areas, confs = [], []
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        areas.append(max(0, x2 - x1) * max(0, y2 - y1) / image_area)
        confs.append(d["confidence"])
    largest = max(areas)
    mean_conf = sum(confs) / len(confs)
    count = len(detections)
    score = min(1.0, 0.45 * min(largest / 0.20, 1.0) + 0.30 * min(count / 5, 1.0) + 0.25 * mean_conf)
    level = "High" if score >= 0.67 else ("Moderate" if score >= 0.34 else "Low")
    return {
        "level": level, "score": round(score, 4),
        "indicators": {"damage_count": count, "largest_bbox_area_ratio": round(largest, 4), "mean_confidence": round(mean_conf, 4)},
        "explanation": "Transparent visual screening heuristic using detection count, confidence and approximate bounding-box area. It is not a certified pavement condition or engineering rating.",
    }
