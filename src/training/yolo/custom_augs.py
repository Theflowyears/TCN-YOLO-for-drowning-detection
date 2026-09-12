import cv2
import numpy as np
from albumentations import DualTransform


class WaterRipple(DualTransform):
    """水面波纹扰动：用正弦位移场 warp 图像，模拟水波折射下目标的形变。

    albumentations 2.x 的坐标约定（本类必须严格遵守）：

    * ``apply()``           收到 uint8 图像；
    * ``apply_to_bboxes()``  收到 **归一化的 xyxy** ``[x_min, y_min, x_max, y_max, *附加列]``，
      并且图像尺寸通过 ``params["shape"] = (H, W, C)`` 传入 ——
      1.x 时代的 ``rows`` / ``cols`` 关键字在 2.x 已不再传递。

    框的位移公式与像素位移完全一致，因此图像内容与标注保持同步。
    """

    def __init__(self,
                 amplitude_range=(0.5, 2.0),
                 frequency_range=(0.01, 0.05),
                 p=0.3):
        # 新版不能传 always_apply，直接省略
        super().__init__(p=p)
        self.amplitude_range = amplitude_range
        self.frequency_range = frequency_range

    @staticmethod
    def _resolve_shape(shape):
        """从 params['shape'] 取 (rows, cols)；拿不到时返回 None。"""
        if shape is None:
            return None
        try:
            dims = tuple(shape)
        except TypeError:
            return None
        if len(dims) >= 2:
            rows, cols = int(dims[0]), int(dims[1])
            if rows > 0 and cols > 0:
                return rows, cols
        return None

    def apply(self, img, amplitude=1.0, frequency=0.02, phase_x=0, phase_y=0, **params):
        h, w = img.shape[:2]
        x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

        dx = amplitude * np.sin(frequency * (x_coords - w / 2) + phase_x)
        dy = amplitude * np.cos(frequency * (y_coords - h / 2) + phase_y)

        map_x = (x_coords - dx).astype(np.float32)
        map_y = (y_coords - dy).astype(np.float32)

        warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT)
        return warped

    def apply_to_bboxes(self, bboxes, amplitude=1.0, frequency=0.02,
                        phase_x=0, phase_y=0, shape=None, **params):
        """对四个角点做同样的位移，再取外接矩形。

        ⚠ 修复说明：旧实现把入参当「像素坐标」处理，并指望框架传 ``rows=`` / ``cols=``。
        albumentations 2.x 传的是归一化 xyxy 加上 ``shape``，于是旧代码里 rows/cols 恒为 0，
        ``np.clip(v, 0, cols - 1)`` 退化为 ``clip(v, 0, -1)``，所有框宽度变成无效被过滤掉 ——
        凡是 WaterRipple 命中的那一半增强图会**静默丢掉全部标注**（等于把目标当背景学）。
        """
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.size == 0:
            return bboxes

        resolved = self._resolve_shape(shape)
        if resolved is None:
            # 拿不到图像尺寸就原样返回：宁可不做这次扰动，也绝不能把标注丢掉
            return bboxes
        rows, cols = resolved

        # 归一化 xyxy -> 像素域，便于与 apply() 共用同一套位移公式
        x_min = bboxes[:, 0] * cols
        y_min = bboxes[:, 1] * rows
        x_max = bboxes[:, 2] * cols
        y_max = bboxes[:, 3] * rows

        corners_x = np.stack([x_min, x_max, x_min, x_max], axis=1)   # (N, 4)
        corners_y = np.stack([y_min, y_min, y_max, y_max], axis=1)

        dx = amplitude * np.sin(frequency * (corners_x - cols / 2) + phase_x)
        dy = amplitude * np.cos(frequency * (corners_y - rows / 2) + phase_y)
        new_x = np.clip(corners_x + dx, 0, cols - 1)
        new_y = np.clip(corners_y + dy, 0, rows - 1)

        # 像素域 -> 归一化
        out_x_min = new_x.min(axis=1) / (cols - 1)
        out_x_max = new_x.max(axis=1) / (cols - 1)
        out_y_min = new_y.min(axis=1) / (rows - 1)
        out_y_max = new_y.max(axis=1) / (rows - 1)

        new_bboxes = bboxes.copy()                 # 保留类别等附加列
        new_bboxes[:, 0], new_bboxes[:, 1] = out_x_min, out_y_min
        new_bboxes[:, 2], new_bboxes[:, 3] = out_x_max, out_y_max

        valid = (new_bboxes[:, 2] > new_bboxes[:, 0]) & (new_bboxes[:, 3] > new_bboxes[:, 1])
        return new_bboxes[valid]

    def apply_to_keypoints(self, keypoints, amplitude=1.0, frequency=0.02,
                           phase_x=0, phase_y=0, shape=None, **params):
        """关键点同样受位移场影响，与图像/标注保持一致。"""
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.size == 0:
            return keypoints
        resolved = self._resolve_shape(shape)
        if resolved is None:
            return keypoints
        rows, cols = resolved

        px = keypoints[:, 0] * cols
        py = keypoints[:, 1] * rows
        dx = amplitude * np.sin(frequency * (px - cols / 2) + phase_x)
        dy = amplitude * np.cos(frequency * (py - rows / 2) + phase_y)

        new_keypoints = keypoints.copy()
        new_keypoints[:, 0] = np.clip(px + dx, 0, cols - 1) / (cols - 1)
        new_keypoints[:, 1] = np.clip(py + dy, 0, rows - 1) / (rows - 1)
        return new_keypoints

    def get_params(self):
        return {
            "amplitude": np.random.uniform(*self.amplitude_range),
            "frequency": np.random.uniform(*self.frequency_range),
            "phase_x": np.random.uniform(0, 2 * np.pi),
            "phase_y": np.random.uniform(0, 2 * np.pi),
        }

    def get_transform_init_args_names(self):
        return ("amplitude_range", "frequency_range")
