import cv2, numpy as np


def overlay(rgb, result):
    out = rgb.copy()
    for g in result.gears:
        color = (
            (255, 60, 40)
            if g.is_defective
            else (40, 210, 80)
            if g.is_defective is False
            else (190, 190, 190)
        )
        out[g.mask] = (out[g.mask] * 0.7 + np.array(color) * 0.3).astype("uint8")
        if g.heatmap is not None:
            heat = cv2.applyColorMap(
                np.uint8(np.clip(g.heatmap, 0, 1) * 255), cv2.COLORMAP_TURBO
            )[:, :, ::-1]
            out[g.mask] = (rgb[g.mask] * 0.55 + heat[g.mask] * 0.45).astype("uint8")
        contours, _ = cv2.findContours(
            g.mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(out, contours, -1, color, 1)
        x, y, _, _ = g.bbox
        # Warm-up is temporary progress, not a third anomaly class.
        text = f"ID{g.gear_id} {g.gear_type or '?'}"
        if g.label is not None:
            text += f" {g.label.upper()}"
        if g.anomaly_score is not None:
            text += f" {g.anomaly_score:.3f}"
        text += f" N:{g.normal_votes} A:{g.anomalous_votes}"
        cv2.putText(
            out,
            text,
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out
