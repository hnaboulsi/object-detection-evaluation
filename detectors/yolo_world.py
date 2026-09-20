"""Adapter for the official AILab-CVC YOLO-World repository."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import Detection, ImageInput, clean_prompts, load_rgb_image, resolve_device

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_WORLD_REPO = REPOSITORY_ROOT / "third_party" / "YOLO-World"
DEFAULT_CONFIG = Path(
    "configs/pretrain/"
    "yolo_world_v2_s_vlpan_bn_2e-3_100e_4x8gpus_"
    "obj365v1_goldg_train_1280ft_lvis_minival.py"
)
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "weights" / "s_stage2-4466ab94.pth"


class YOLOWorld:
    def __init__(
        self,
        device: str = "auto",
        repo_path: str | Path | None = None,
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        score_threshold: float = 0.25,
        max_detections: int = 100,
    ) -> None:
        resolved_device = resolve_device(device)
        if resolved_device == "mps":
            raise RuntimeError(
                "Official YOLO-World/MMCV is not supported on MPS here; use --device cpu or CUDA"
            )
        try:
            import torch
            import yolo_world  # noqa: F401  # registers official model classes
            from mmdet.apis import init_detector
            from mmdet.utils import get_test_pipeline_cfg
            from mmengine.config import Config
            from mmengine.dataset import Compose
        except ImportError as error:
            raise RuntimeError(
                "Official YOLO-World/MMCV could not load. Use the isolated setup in the README; if MMCV native ops fail on Apple Silicon, run it on Linux/CUDA."
            ) from error

        self._torch = torch
        self._Config = Config
        self._Compose = Compose
        self._init_detector = init_detector
        self._get_test_pipeline_cfg = get_test_pipeline_cfg
        self.device = resolved_device
        self.repo_path = (
            Path(
                repo_path or os.environ.get("YOLO_WORLD_REPO", DEFAULT_YOLO_WORLD_REPO)
            )
            .expanduser()
            .resolve()
        )
        self.config_path = Path(
            config_path or self.repo_path / DEFAULT_CONFIG
        ).resolve()
        self.checkpoint_path = (
            Path(
                checkpoint_path
                or os.environ.get("YOLO_WORLD_CHECKPOINT", DEFAULT_CHECKPOINT)
            )
            .expanduser()
            .resolve()
        )
        if not self.config_path.is_file():
            raise RuntimeError(
                f"YOLO-World config not found: {self.config_path}. Clone the official repository as described in the README."
            )
        if not self.checkpoint_path.is_file():
            raise RuntimeError(
                f"YOLO-World checkpoint not found: {self.checkpoint_path}. Download the official V2.1-S checkpoint."
            )
        self.score_threshold = float(score_threshold)
        self.max_detections = int(max_detections)
        self.model_name = "AILab-CVC/YOLO-World-V2.1-S"

        cfg = self._Config.fromfile(str(self.config_path))
        cfg.load_from = str(self.checkpoint_path)
        device_name = "cuda:0" if self.device == "cuda" else "cpu"
        self.model = self._init_detector(
            cfg, checkpoint=str(self.checkpoint_path), device=device_name
        )
        pipeline_cfg = self._get_test_pipeline_cfg(cfg=cfg)
        pipeline_cfg[0].type = "mmdet.LoadImageFromNDArray"
        self.pipeline = self._Compose(pipeline_cfg)

    def detect(self, image: ImageInput, prompts: Sequence[str]) -> list[Detection]:
        prompt_list = clean_prompts(prompts)
        texts = [[prompt] for prompt in prompt_list] + [[" "]]
        self.model.reparameterize(texts)
        rgb = np.asarray(load_rgb_image(image))
        data_info = self.pipeline({"img": rgb, "img_id": 0, "texts": texts})
        batch = {
            "inputs": data_info["inputs"].unsqueeze(0),
            "data_samples": [data_info["data_samples"]],
        }
        with self._torch.no_grad():
            output = self.model.test_step(batch)[0]
        instances = output.pred_instances
        instances = instances[instances.scores.float() > self.score_threshold]
        if len(instances.scores) > self.max_detections:
            keep = instances.scores.float().topk(self.max_detections)[1]
            instances = instances[keep]
        values = instances.cpu().numpy()
        detections: list[Detection] = []
        for box, label_index, score in zip(
            values["bboxes"], values["labels"], values["scores"]
        ):
            index = int(label_index)
            if index >= len(prompt_list):
                continue
            detections.append(
                Detection(
                    label=prompt_list[index],
                    score=float(score),
                    bbox=tuple(float(value) for value in box.tolist()),
                    model_name=self.model_name,
                )
            )
        return detections

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "yolo_world",
            "model_name": self.model_name,
            "device": self.device,
            "score_threshold": self.score_threshold,
            "config_path": str(self.config_path),
            "checkpoint_path": str(self.checkpoint_path),
        }
