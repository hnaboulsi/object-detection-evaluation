"""Grounding DINO through Hugging Face Transformers."""

from __future__ import annotations

from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any

from artifacts import file_hash

from .base import Detection, ImageInput, clean_prompts, load_rgb_image, resolve_device


class GroundingDINO:
    DEFAULT_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"

    def __init__(
        self,
        device: str = "auto",
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        revision: str = DEFAULT_REVISION,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Grounding DINO requires the packages in requirements.txt"
            ) from error

        self._torch = torch
        self.model_name = model_name
        self.revision = revision
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        if not 0 <= self.box_threshold <= 1 or not 0 <= self.text_threshold <= 1:
            raise ValueError("Detection thresholds must be between zero and one")
        self.device = resolve_device(device)
        self.processor = AutoProcessor.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_name, revision=revision
        )
        self.model.to(self.device).eval()
        from huggingface_hub import hf_hub_download

        weights = hf_hub_download(model_name, "model.safetensors", revision=revision)
        self.checkpoint_sha256 = file_hash(Path(weights))

    def detect(self, image: ImageInput, prompts: Sequence[str]) -> list[Detection]:
        prompt_list = clean_prompts(prompts)
        pil_image = load_rgb_image(image)
        inputs = self.processor(
            images=pil_image,
            text=[prompt_list],
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],
            text_labels=[prompt_list],
        )[0]
        return [
            Detection(
                label=str(label),
                score=float(score.item()),
                bbox=tuple(float(value) for value in box.tolist()),
                model_name=self.model_name,
            )
            for box, score, label in zip(
                result["boxes"], result["scores"], result["text_labels"], strict=True
            )
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "grounding_dino",
            "model_name": self.model_name,
            "revision": self.revision,
            "device": self.device,
            "box_threshold": self.box_threshold,
            "text_threshold": self.text_threshold,
            "checkpoint_sha256": self.checkpoint_sha256,
            "torch_version": version("torch"),
            "transformers_version": version("transformers"),
            "torch_threads": self._torch.get_num_threads(),
        }
