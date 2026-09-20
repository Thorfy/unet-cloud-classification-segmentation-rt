"""Instances de nuages a partir du masque semantique (composantes connexes)."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.cloud_types import CLASSES, CLOUD_IDS, id_to_name, palette


def mask_to_instances(
    mask: np.ndarray,
    min_area: int | None = None,
    max_instances: int = 80,
) -> list[dict[str, Any]]:
    arr = np.asarray(mask)
    h, w = arr.shape[:2]
    if min_area is None:
        min_area = max(64, int(0.00025 * h * w))
    items: list[dict[str, Any]] = []
    for class_id in CLOUD_IDS:
        binary = (arr == class_id).astype(np.uint8)
        if not binary.any():
            continue
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            bw = int(stats[idx, cv2.CC_STAT_WIDTH])
            bh = int(stats[idx, cv2.CC_STAT_HEIGHT])
            items.append(
                {
                    "class_id": int(class_id),
                    "class_key": CLASSES[class_id][0],
                    "label": id_to_name(class_id),
                    "area_px": area,
                    "coverage": float(area) / float(h * w),
                    "bbox": (x, y, x + bw, y + bh),
                    "component": int(idx),
                }
            )
    items.sort(key=lambda it: it["area_px"], reverse=True)
    for i, item in enumerate(items, start=1):
        item["id"] = i
    return items[:max_instances]


def draw_instance_boxes(canvas: np.ndarray, instances: list[dict[str, Any]]) -> np.ndarray:
    """Dessine les boites d'instance sur une image RGB deja composee."""
    out = np.asarray(canvas).copy()
    colors = palette()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for item in instances:
        x1, y1, x2, y2 = item["bbox"]
        color = tuple(int(c) for c in colors[item["class_id"]])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        caption = f"#{item['id']} {item['label'].split(' (')[0]}"
        (tw, th), _ = cv2.getTextSize(caption, font, 0.45, 1)
        ty = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, ty), (x1 + tw + 6, ty + th + 6), color, -1)
        cv2.putText(out, caption, (x1 + 3, ty + th + 2), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def annotate_instances(rgb: np.ndarray, mask: np.ndarray, instances: list[dict[str, Any]]) -> np.ndarray:
    """Overlay semantique + boites d'instance numerotees."""
    from src.cloud_types import overlay_types

    canvas = overlay_types(rgb, mask, opacity=0.40)
    return draw_instance_boxes(canvas, instances)


def instances_table(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in instances:
        x1, y1, x2, y2 = item["bbox"]
        rows.append(
            {
                "id": item["id"],
                "type": item["label"],
                "couverture_%": round(item["coverage"] * 100, 2),
                "pixels": item["area_px"],
                "bbox": f"{x1},{y1},{x2},{y2}",
            }
        )
    return rows
