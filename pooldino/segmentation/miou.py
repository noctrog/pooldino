"""Streaming confusion matrix + mIoU helpers for semantic segmentation."""

from __future__ import annotations

import numpy as np


class ConfusionMatrix:
    """Streaming confusion matrix for semantic segmentation.

    Accumulates (target, pred) pixel counts as a ``(num_classes, num_classes)``
    int64 matrix. Pixels equal to ``ignore_index`` are excluded.

    Use :meth:`update` for each batch, :meth:`summarize` at the end.
    """

    def __init__(self, num_classes: int):
        assert num_classes > 0
        self.n = num_classes
        self.cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.cm.fill(0)

    def update(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        *,
        ignore_index: int = -1,
    ) -> None:
        """Accumulate a batch of predictions.

        Args:
            preds: int array of predicted class indices, any shape.
            targets: int array of target class indices, same shape as ``preds``.
            ignore_index: target value that should be excluded from the matrix.
        """
        if preds.shape != targets.shape:
            raise ValueError(f"preds shape {preds.shape} != targets shape {targets.shape}")
        p = preds.ravel().astype(np.int64, copy=False)
        t = targets.ravel().astype(np.int64, copy=False)
        mask = t != ignore_index
        # Also guard against out-of-range predictions — argmax should always be
        # in range for a softmax head, but keep it defensive for eval-time safety.
        mask &= (p >= 0) & (p < self.n) & (t >= 0) & (t < self.n)
        if not mask.any():
            return
        p = p[mask]
        t = t[mask]
        idx = t * self.n + p
        bincount = np.bincount(idx, minlength=self.n * self.n)
        self.cm += bincount.reshape(self.n, self.n)

    def summarize(self) -> dict:
        """Return per-class IoU, mIoU, per-class accuracy, mAcc, pixel accuracy.

        NaN entries in ``per_class_iou`` / ``per_class_acc`` mark classes that
        never appeared in ground truth or were never predicted. ``miou`` and
        ``macc`` ignore those NaN entries.
        """
        cm = self.cm
        tp = np.diag(cm).astype(np.float64)
        fp = cm.sum(axis=0).astype(np.float64) - tp
        fn = cm.sum(axis=1).astype(np.float64) - tp
        union = tp + fp + fn
        row_sum = cm.sum(axis=1).astype(np.float64)

        with np.errstate(invalid="ignore", divide="ignore"):
            per_class_iou = np.where(union > 0, tp / np.maximum(union, 1), np.nan)
            per_class_acc = np.where(row_sum > 0, tp / np.maximum(row_sum, 1), np.nan)
        total = cm.sum()
        pixel_acc = float(tp.sum() / total) if total > 0 else 0.0

        return {
            "miou": float(np.nanmean(per_class_iou)) if np.any(union > 0) else 0.0,
            "macc": float(np.nanmean(per_class_acc)) if np.any(row_sum > 0) else 0.0,
            "pixel_acc": pixel_acc,
            "per_class_iou": per_class_iou.tolist(),
            "per_class_acc": per_class_acc.tolist(),
        }


def _reference_intersect_union(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Reference per-class ``(intersection, union)`` computation.

    Mirrors mmsegmentation's ``IoUMetric.intersect_and_union`` formulation
    (a different code path from our confusion-matrix accumulator). Useful as
    a cross-check for numerical agreement.
    """
    valid = target != ignore_index
    p = pred[valid].astype(np.int64)
    t = target[valid].astype(np.int64)
    # Intersection: pixels where pred == target, bucketed by class.
    eq = p == t
    intersect = np.bincount(p[eq], minlength=num_classes)[:num_classes]
    area_pred = np.bincount(p, minlength=num_classes)[:num_classes]
    area_target = np.bincount(t, minlength=num_classes)[:num_classes]
    union = area_pred + area_target - intersect
    return intersect.astype(np.float64), union.astype(np.float64)


def _run_cross_check() -> None:
    """Fuzz our ConfusionMatrix against the intersect/union reference."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        num_classes = int(rng.integers(2, 30))
        h, w = int(rng.integers(8, 64)), int(rng.integers(8, 64))
        b = int(rng.integers(1, 4))
        target = rng.integers(-1, num_classes, size=(b, h, w), dtype=np.int64)
        pred = rng.integers(0, num_classes, size=(b, h, w), dtype=np.int64)

        cm = ConfusionMatrix(num_classes)
        cm.update(pred, target, ignore_index=-1)
        ours = cm.summarize()

        intersect, union = _reference_intersect_union(pred, target, num_classes, -1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ref_per_class = np.where(union > 0, intersect / np.maximum(union, 1), np.nan)
        ref_miou = float(np.nanmean(ref_per_class)) if np.any(union > 0) else 0.0

        assert abs(ours["miou"] - ref_miou) < 1e-9, (
            f"trial {trial}: ours={ours['miou']!r} ref={ref_miou!r}"
        )
        ours_iou = np.asarray(ours["per_class_iou"], dtype=np.float64)
        # NaN-safe equality
        mask = ~(np.isnan(ours_iou) | np.isnan(ref_per_class))
        assert np.allclose(ours_iou[mask], ref_per_class[mask], atol=1e-9), (
            f"trial {trial}: per-class disagreement"
        )
    print("miou cross-check: 20 random trials PASSED")


if __name__ == "__main__":
    _run_cross_check()
