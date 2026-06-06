from __future__ import annotations

import gc
import logging
from typing import Optional, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticSegmentatorProtocol(Protocol):
    def load(self) -> None:
        ...

    def close(self) -> None:
        ...

    def predict_semantic01(self, x: torch.Tensor, inference_params=None) -> np.ndarray:
        ...


class UNetDown(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_rep: int,
        batch_norm: bool = False,
        activation: nn.Module = nn.ReLU(),
        kernel_size: int = 3,
        dropout: bool = False,
        maxpool: bool = False,
    ):
        super().__init__()
        self.down_block = nn.Sequential()

        if maxpool:
            self.down_block.add_module("maxpool", nn.MaxPool2d(2))

        in_ch_for_conv = in_ch
        for k in range(num_rep):
            self.down_block.add_module(
                f"conv{k+1}",
                nn.Conv2d(
                    in_ch_for_conv,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
            )
            self.down_block.add_module(f"act{k+1}", activation)
            if batch_norm:
                self.down_block.add_module(f"bn{k+1}", nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch

        if dropout:
            self.down_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.down_block(inp)


class UNetUp(nn.Module):
    def __init__(
        self,
        in_ch: int,
        res_ch: int,
        out_ch: int,
        num_rep: int,
        batch_norm: bool = False,
        activation: nn.Module = nn.ReLU(),
        kernel_size: int = 3,
        dropout: bool = False,
    ):
        super().__init__()
        self.up = nn.Sequential()
        self.conv_block = nn.Sequential()

        self.up.add_module(
            "conv2d_transpose",
            nn.ConvTranspose2d(
                in_ch,
                in_ch,
                kernel_size,
                stride=2,
                output_padding=(kernel_size - 1) // 2,
                padding=(kernel_size - 1) // 2,
            ),
        )
        if batch_norm:
            self.up.add_module("bn1", nn.BatchNorm2d(in_ch))

        in_ch_for_conv = in_ch + res_ch
        for k in range(num_rep):
            self.conv_block.add_module(
                f"conv{k+1}",
                nn.Conv2d(
                    in_ch_for_conv,
                    out_ch,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
            )
            self.conv_block.add_module(f"act{k+1}", activation)
            if batch_norm:
                self.conv_block.add_module(f"bn{k+2}", nn.BatchNorm2d(out_ch))
            in_ch_for_conv = out_ch

        if dropout:
            self.conv_block.add_module("dropout", nn.Dropout2d(p=0.5))

    def forward(self, inp: torch.Tensor, res: Optional[torch.Tensor] = None) -> torch.Tensor:
        feat = self.up(inp)
        merged = feat if res is None else torch.cat([feat, res], dim=1)
        return self.conv_block(merged)


class ConvSig(nn.Module):
    """ Conv layer + Sigmoid

    Args:
        in_ch (int): Number of input channels
    """

    def __init__(self, in_ch):
        super(ConvSig, self).__init__()
        self.out = nn.Sequential()
        self.out.add_module("conv2d", nn.Conv2d(in_ch, 1, 1))
        self.out.add_module("sigmoid", nn.Sigmoid())

    def forward(self, inp):
        return self.out(inp)


class unet_vgg16(nn.Module):
    def __init__(self, inp_ch: int, kernel_size: int = 3, skip: bool = True):
        super().__init__()
        self.skip = bool(skip)

        self.enc1 = UNetDown(inp_ch, 64, 2, batch_norm=True, maxpool=False, kernel_size=kernel_size)
        self.enc2 = UNetDown(64, 128, 2, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc3 = UNetDown(128, 256, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc4 = UNetDown(256, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)
        self.enc5 = UNetDown(512, 512, 3, batch_norm=True, maxpool=True, kernel_size=kernel_size)

        self.dec4 = UNetUp(512, 512 if self.skip else 0, 512, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec3 = UNetUp(512, 256 if self.skip else 0, 256, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec2 = UNetUp(256, 128 if self.skip else 0, 128, 2, batch_norm=True, kernel_size=kernel_size)
        self.dec1 = UNetUp(128, 64 if self.skip else 0, 64, 2, batch_norm=True, kernel_size=kernel_size)

        self.out = ConvSig(64)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        d1 = self.enc1(inp)
        d2 = self.enc2(d1)
        d3 = self.enc3(d2)
        d4 = self.enc4(d3)
        d5 = self.enc5(d4)

        if self.skip:
            u4 = self.dec4(d5, d4)
            u3 = self.dec3(u4, d3)
            u2 = self.dec2(u3, d2)
            u1 = self.dec1(u2, d1)
        else:
            u4 = self.dec4(d5, None)
            u3 = self.dec3(u4, None)
            u2 = self.dec2(u3, None)
            u1 = self.dec1(u2, None)

        return self.out(u1)


class UNetVGG16Segmentator:
    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        half: bool = False,
        threshold: float = 0.5,
        skip: bool = True,
        inp_ch: int = 9,
        logger: logging.Logger | None = None,
    ):
        self._model: Optional[nn.Module] = None
        self._device = str(device)
        self._weights_path = str(weights_path)
        self.half = bool(half)
        self.threshold = float(threshold)
        self.skip = bool(skip)
        self.inp_ch = int(inp_ch)
        self._logger = logger or logging.getLogger("UNetVGG16Segmentator")

    @staticmethod
    def resolve_device(device: str) -> str:
        device = str(device).strip().lower()
        if device in ("auto", ""):
            if torch.cuda.is_available():
                return "cuda:0"
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            return "cpu"

        if device.startswith("cuda"):
            if torch.cuda.is_available():
                return device if ":" in device else "cuda:0"
            return "cpu"

        if device.startswith("mps"):
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend is not None and mps_backend.is_available():
                return "mps"
            return "cpu"

        return device

    def build_model(self) -> nn.Module:
        return unet_vgg16(inp_ch=self.inp_ch, skip=self.skip)

    def load(self) -> None:
        requested = self._device
        self._device = self.resolve_device(requested)
        if self._device != requested:
            self._logger.warning(
                f"[CD-MODEL] device={requested!r} unavailable, using {self._device!r}"
            )
        if self.half and "cuda" not in self._device:
            self._logger.warning("[CD-MODEL] half precision requires CUDA, disabled")
            self.half = False

        model = self.build_model()

        ckpt = torch.load(self._weights_path, map_location="cpu", weights_only=True)

        if not isinstance(ckpt, dict):
            raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt

        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = model.load_state_dict(state, strict=True)

        model.eval().to(self._device)
        if self.half and "cuda" in self._device:
            model.half()

        self._model = model
        self._logger.info(
            f"[CD-MODEL] loaded weights={self._weights_path} "
            f"device={self._device} inp_ch={self.inp_ch} half={self.half} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, mult: int = 16):
        _, _, h, w = x.shape
        ph = (mult - (h % mult)) % mult
        pw = (mult - (w % mult)) % mult
        if ph == 0 and pw == 0:
            return x, (0, 0, 0, 0)
        pad = (0, pw, 0, ph)
        return F.pad(x, pad, mode="reflect"), pad

    @staticmethod
    def _unpad(y: torch.Tensor, pad):
        if pad == (0, 0, 0, 0):
            return y
        left, right, top, bottom = pad
        h = y.shape[-2] - top - bottom
        w = y.shape[-1] - left - right
        return y[..., top:top + h, left:left + w]

    @torch.no_grad()
    def predict_semantic01(self, x: torch.Tensor, inference_params=None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call load() first")

        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")

        x = x.to(self._device, non_blocking=True)
        x = x.half() if (self.half and "cuda" in self._device) else x.float()

        x_pad, pad = self._pad_to_multiple(x, mult=16)
        prob = self._model(x_pad)
        prob = self._unpad(prob, pad)

        prob_np = prob[0, 0].float().cpu().numpy()
        sem01 = (prob_np >= self.threshold).astype(np.float32)
        return sem01

    def release_memory(self):
        if self._model is not None:
            del self._model
        self._model = None
        gc.collect()
        if self._device and "cuda" in self._device:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def close(self) -> None:
        self.release_memory()