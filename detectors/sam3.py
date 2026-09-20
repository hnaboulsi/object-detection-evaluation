"""Text-prompted detection using Meta's official SAM 3 image processor."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import Detection, ImageInput, clean_prompts, load_rgb_image, resolve_device


class SAM3:
    def __init__(
        self,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        checkpoint_path: str | None = None,
    ) -> None:
        resolved_device = resolve_device(device)
        if resolved_device != "cuda":
            raise RuntimeError(
                "Official SAM 3 requires CUDA. Install requirements-sam3.txt "
                "in a supported NVIDIA environment."
            )
        try:
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise RuntimeError(
                "SAM 3 is not installed. Use a separate Python 3.12/CUDA environment "
                "with requirements-sam3.txt."
            ) from error

        self.device = resolved_device
        self.confidence_threshold = float(confidence_threshold)
        self.model_name = "facebook/sam3"
        try:
            self.model = build_sam3_image_model(
                device=self.device,
                checkpoint_path=checkpoint_path,
                load_from_HF=checkpoint_path is None,
            )
            self.processor = Sam3Processor(
                self.model,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
        except Exception as error:
            raise RuntimeError(
                "SAM 3 could not initialize. Check the underlying error, CUDA compatibility, "
                "and access to its gated Hugging Face checkpoint."
            ) from error

    def detect(self, image: ImageInput, prompts: Sequence[str]) -> list[Detection]:
        prompt_list = clean_prompts(prompts)
        pil_image = load_rgb_image(image)
        detections: list[Detection] = []
        for prompt in prompt_list:
            state = self.processor.set_image(pil_image)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            masks = output["masks"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            for mask, score in zip(masks, scores, strict=True):
                if float(score) < self.confidence_threshold:
                    continue
                pixels = np.asarray(mask).squeeze().astype(bool)
                rows, columns = np.nonzero(pixels)
                if len(columns) == 0:
                    continue
                detections.append(
                    Detection(
                        label=prompt,
                        score=float(score),
                        bbox=(
                            float(columns.min()),
                            float(rows.min()),
                            float(columns.max() + 1),
                            float(rows.max() + 1),
                        ),
                        model_name=self.model_name,
                    )
                )
        return detections

    def metadata(self) -> dict[str, Any]:
        return {
            "name": "sam3",
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "boxes_derived_from_masks": True,
        }
