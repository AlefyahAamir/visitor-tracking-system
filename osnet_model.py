"""Standalone OSNet x0.25 model used for person re-identification.

The architecture is based on the MIT-licensed Torchreid OSNet implementation:
https://github.com/KaiyangZhou/deep-person-reid

Only the inference architecture and a small compatible weight loader are kept
here, so VisitorTrackingSystem does not depend on importing the full
``torchreid`` package at runtime.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class ConvLayer(nn.Module):
    """Convolution + normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        use_instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        if use_instance_norm:
            self.bn: nn.Module = nn.InstanceNorm2d(out_channels, affine=True)
        else:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    """1x1 convolution + batch normalization + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    """1x1 convolution + batch normalization without activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Pointwise linear convolution followed by depthwise 3x3 convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            bias=False,
            groups=out_channels,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.relu(self.bn(x))


class ChannelGate(nn.Module):
    """Generate channel-wise gates conditioned on the input tensor."""

    def __init__(
        self,
        in_channels: int,
        num_gates: int | None = None,
        return_gates: bool = False,
        gate_activation: str = "sigmoid",
        reduction: int = 16,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_gates is None:
            num_gates = in_channels

        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(1, in_channels // reduction)
        self.fc1 = nn.Conv2d(
            in_channels,
            reduced_channels,
            kernel_size=1,
            bias=True,
            padding=0,
        )
        self.norm1: nn.Module | None = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((reduced_channels, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduced_channels,
            num_gates,
            kernel_size=1,
            bias=True,
            padding=0,
        )

        if gate_activation == "sigmoid":
            self.gate_activation: nn.Module | None = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_tensor = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input_tensor * x


class OSBlock(nn.Module):
    """Omni-scale feature-learning block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_instance_norm: bool = False,
        bottleneck_reduction: int = 4,
    ) -> None:
        super().__init__()
        mid_channels = out_channels // bottleneck_reduction

        self.conv1 = Conv1x1(in_channels, mid_channels)
        self.conv2a = LightConv3x3(mid_channels, mid_channels)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
            LightConv3x3(mid_channels, mid_channels),
        )
        self.gate = ChannelGate(mid_channels)
        self.conv3 = Conv1x1Linear(mid_channels, out_channels)

        self.downsample: nn.Module | None = None
        if in_channels != out_channels:
            self.downsample = Conv1x1Linear(in_channels, out_channels)

        self.instance_norm: nn.Module | None = None
        if use_instance_norm:
            self.instance_norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = (
            self.gate(x2a)
            + self.gate(x2b)
            + self.gate(x2c)
            + self.gate(x2d)
        )
        x3 = self.conv3(x2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        output = x3 + identity
        if self.instance_norm is not None:
            output = self.instance_norm(output)
        return F.relu(output)


class OSNet(nn.Module):
    """Omni-Scale Network for person re-identification."""

    def __init__(
        self,
        num_classes: int,
        layers: tuple[int, int, int],
        channels: tuple[int, int, int, int],
        feature_dim: int = 512,
        loss: str = "softmax",
        use_instance_norm: bool = False,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(
            3,
            channels[0],
            7,
            stride=2,
            padding=3,
            use_instance_norm=use_instance_norm,
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(
            layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
            use_instance_norm=use_instance_norm,
        )
        self.conv3 = self._make_layer(
            layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True,
        )
        self.conv4 = self._make_layer(
            layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False,
        )
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer(feature_dim, channels[3])
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(
        layer_count: int,
        in_channels: int,
        out_channels: int,
        reduce_spatial_size: bool,
        use_instance_norm: bool = False,
    ) -> nn.Sequential:
        modules: list[nn.Module] = [
            OSBlock(
                in_channels,
                out_channels,
                use_instance_norm=use_instance_norm,
            )
        ]
        for _ in range(1, layer_count):
            modules.append(
                OSBlock(
                    out_channels,
                    out_channels,
                    use_instance_norm=use_instance_norm,
                )
            )

        if reduce_spatial_size:
            modules.append(
                nn.Sequential(
                    Conv1x1(out_channels, out_channels),
                    nn.AvgPool2d(2, stride=2),
                )
            )
        return nn.Sequential(*modules)

    def _construct_fc_layer(
        self,
        fc_dims: int | list[int] | None,
        input_dim: int,
        dropout_p: float | None = None,
    ) -> nn.Sequential | None:
        if fc_dims is None or (isinstance(fc_dims, int) and fc_dims < 0):
            self.feature_dim = input_dim
            return None
        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        modules: list[nn.Module] = []
        assert fc_dims is not None
        for dimension in fc_dims:
            modules.append(nn.Linear(input_dim, dimension))
            modules.append(nn.BatchNorm1d(dimension))
            modules.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                modules.append(nn.Dropout(p=dropout_p))
            input_dim = dimension

        self.feature_dim = fc_dims[-1]
        return nn.Sequential(*modules)

    def _init_params(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def featuremaps(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)

    def forward(
        self,
        x: torch.Tensor,
        return_featuremaps: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.featuremaps(x)
        if return_featuremaps:
            return x

        features = self.global_avgpool(x)
        features = features.view(features.size(0), -1)
        if self.fc is not None:
            features = self.fc(features)

        if not self.training:
            return features

        logits = self.classifier(features)
        if self.loss == "softmax":
            return logits
        if self.loss == "triplet":
            return logits, features
        raise KeyError(f"Unsupported loss: {self.loss}")


def build_osnet_x0_25(
    num_classes: int = 1000,
    loss: str = "softmax",
) -> OSNet:
    """Build the OSNet x0.25 architecture without downloading weights."""

    return OSNet(
        num_classes=num_classes,
        layers=(2, 2, 2),
        channels=(16, 64, 96, 128),
        loss=loss,
    )


def _load_checkpoint(path: Path, map_location: str | torch.device) -> Any:
    """Load a checkpoint across supported PyTorch versions."""

    try:
        return torch.load(
            str(path),
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        return torch.load(str(path), map_location=map_location)
    except Exception:
        # Some older Torchreid checkpoints need the legacy loader behavior.
        return torch.load(
            str(path),
            map_location=map_location,
            weights_only=False,
        )


def load_pretrained_weights(
    model: nn.Module,
    weight_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    """Load compatible checkpoint layers and ignore classifier mismatches."""

    path = Path(weight_path)
    if not path.is_file():
        raise FileNotFoundError(f"OSNet weight file was not found: {path}")

    checkpoint = _load_checkpoint(path, map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise RuntimeError("The OSNet checkpoint does not contain a state dictionary.")

    model_dict = model.state_dict()
    compatible = OrderedDict()
    discarded = 0

    for key, value in state_dict.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if (
            clean_key in model_dict
            and hasattr(value, "size")
            and model_dict[clean_key].size() == value.size()
        ):
            compatible[clean_key] = value
        else:
            discarded += 1

    if not compatible:
        raise RuntimeError(
            "No OSNet checkpoint layers matched the local OSNet architecture."
        )

    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    return len(compatible), discarded
