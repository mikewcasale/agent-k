"""Offline ResNet18 training and inference for AGENT-K vision tasks.

(c) Mike Casale 2025.
Licensed under the MIT License.
"""

from __future__ import annotations as _annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim.swa_utils import update_bn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18

from .datasets import TaskKind
from .evaluator import (
    CandidateEngine,
    CandidateResult,
    CandidateSpec,
    load_candidate_dataset,
    write_candidate_predictions,
)
from .metrics import MetricId, PredictionKind, metric_spec, prediction_columns

__all__ = ("run_candidate",)

_VISION_PARAMETERS: Final[frozenset[str]] = frozenset(
    {
        "batch_size",
        "cpu_threads",
        "device_policy",
        "epochs",
        "freeze_backbone",
        "image_size",
        "learning_rate",
        "weight_decay",
    }
)


class _VisionConfig(BaseModel):
    """Validated ResNet18 training controls."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=32, ge=1, le=512)
    cpu_threads: int = Field(default=1, ge=1, le=64)
    device_policy: str = "auto"
    epochs: int = Field(default=5, ge=1, le=100)
    freeze_backbone: bool = False
    image_size: int = Field(default=224, ge=32, le=1_024)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)


class _ImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load real image assets lazily from a candidate-visible table."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_column: str,
        image_root: str,
        input_root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        target: np.ndarray[Any, Any] | None,
    ) -> None:
        self._paths = [
            _safe_image_path(input_root, str(Path(image_root) / relative))
            for relative in frame[image_column].astype(str)
        ]
        self._transform = transform
        self._target = target

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self._paths[index]) as image:
            tensor = self._transform(image.convert("RGB"))
        if self._target is None:
            return tensor, torch.tensor(0)
        return tensor, torch.as_tensor(self._target[index])


def run_candidate(
    spec: CandidateSpec,
    input_root: Path,
    output_root: Path,
) -> CandidateResult:
    """Fine-tune an actual ResNet18 and emit trusted-contract predictions."""
    started = time.monotonic()
    if spec.engine is not CandidateEngine.RESNET18:
        raise ValueError(f"unsupported vision engine: {spec.engine.value}")
    if spec.image_column is None:
        raise ValueError("vision candidates require image_column")
    if len(spec.target_columns) != 1:
        raise ValueError("vision candidates require exactly one target")
    unknown = set(spec.hyperparameters) - _VISION_PARAMETERS
    if unknown:
        raise ValueError(f"unsupported ResNet18 hyperparameters: {sorted(unknown)}")

    config = _VisionConfig.model_validate(spec.hyperparameters)
    bundle = load_candidate_dataset(spec, input_root)
    seed = spec.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _select_device(config.device_policy)
    if device.type == "cpu":
        torch.set_num_threads(config.cpu_threads)

    transform = transforms.Compose(
        (
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        )
    )
    target, output_count = _vision_target(spec, bundle.train_targets.iloc[:, 0])
    train_dataset = _ImageDataset(
        bundle.train,
        image_column=spec.image_column,
        image_root=spec.train_image_root or "",
        input_root=input_root,
        transform=transform,
        target=target,
    )
    predict_dataset = _ImageDataset(
        bundle.predict,
        image_column=spec.image_column,
        image_root=spec.predict_image_root or "",
        input_root=input_root,
        transform=transform,
        target=None,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    predict_loader = DataLoader(
        predict_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = resnet18(weights=None)
    _load_resnet_weights(model, spec, input_root)
    if config.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, output_count)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion: nn.Module = (
        nn.CrossEntropyLoss() if spec.problem_type.task is TaskKind.CLASSIFICATION else nn.MSELoss()
    )
    model.train()
    for _ in range(config.epochs):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            if spec.problem_type.task is TaskKind.REGRESSION:
                labels = labels.to(torch.float32).reshape(-1, 1)
            loss = criterion(output, labels)
            if not torch.isfinite(loss):
                raise ValueError("ResNet18 training produced a non-finite loss")
            loss.backward()
            optimizer.step()

    update_bn(train_loader, model, device=device)
    calibration_batches = int(model.bn1.num_batches_tracked.item())
    raw = _predict(model, predict_loader, device)
    predictions = _vision_predictions(spec, raw)
    return write_candidate_predictions(
        spec,
        bundle.predict_ids,
        predictions,
        output_root,
        started=started,
        diagnostics={
            "engine": CandidateEngine.RESNET18.value,
            "device": device.type,
            "epochs": config.epochs,
            "image_size": config.image_size,
            "batch_norm_calibrated": True,
            "batch_norm_calibration_batches": calibration_batches,
        },
    )


def _vision_target(
    spec: CandidateSpec,
    target: pd.Series,
) -> tuple[np.ndarray[Any, Any], int]:
    if spec.problem_type.task is TaskKind.REGRESSION:
        numeric = pd.to_numeric(target, errors="raise").to_numpy(dtype=np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("vision regression target contains non-finite values")
        return numeric, 1
    labels = target.astype(str)
    if set(labels) != set(spec.class_labels):
        raise ValueError("vision training classes differ from CandidateSpec.class_labels")
    encoding = {label: index for index, label in enumerate(spec.class_labels)}
    return labels.map(encoding).to_numpy(dtype=np.int64), len(spec.class_labels)


def _predict(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> np.ndarray[Any, Any]:
    batches: list[np.ndarray[Any, Any]] = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            batches.append(model(images.to(device)).cpu().numpy())
    return np.concatenate(batches, axis=0)


def _vision_predictions(
    spec: CandidateSpec,
    raw: np.ndarray[Any, Any],
) -> pd.DataFrame:
    columns = prediction_columns(spec.metric_id, spec.target_columns, spec.class_labels)
    if spec.problem_type.task is TaskKind.REGRESSION:
        prediction = raw.reshape(-1).astype(np.float64)
        if spec.metric_id is MetricId.RMSLE:
            prediction = np.maximum(prediction, 0.0)
        return pd.DataFrame({columns[0]: prediction})
    shifted = raw - raw.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    if metric_spec(spec.metric_id).prediction_kind is PredictionKind.PROBABILITY:
        if len(spec.class_labels) == 2:
            return pd.DataFrame({columns[0]: probabilities[:, 1]})
        return pd.DataFrame(
            {column: probabilities[:, index] for index, column in enumerate(columns)}
        )
    hard = np.argmax(probabilities, axis=1)
    return pd.DataFrame({columns[0]: [spec.class_labels[index] for index in hard]})


def _load_resnet_weights(
    model: nn.Module,
    spec: CandidateSpec,
    input_root: Path,
) -> None:
    relative = spec.offline_assets.get("resnet18_weights")
    if relative is None:
        return
    path = _safe_image_path(input_root, relative)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("ResNet18 weight artifact is not a state dictionary")
    model.load_state_dict(state)


def _select_device(policy: str) -> torch.device:
    if policy not in {"auto", "cuda_required", "cpu"}:
        raise ValueError("device_policy must be auto, cuda_required, or cpu")
    if policy == "cuda_required" and not torch.cuda.is_available():
        raise RuntimeError("vision candidate requires CUDA but no CUDA device is available")
    if policy == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _safe_image_path(root: Path, relative_path: str) -> Path:
    path = (root.resolve() / relative_path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"image path escapes input root: {relative_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"image asset does not exist: {relative_path}")
    return path
