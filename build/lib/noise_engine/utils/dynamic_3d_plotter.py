import logging

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, ClassVar, Optional, Any


class PlotMode(Enum):
    SCATTER = "scatter"
    SURFACE = "surface"
    VOXEL = "voxel"
    BAR = "bar"


@dataclass
class Dynamic3DPlotter:
    """Utility for dynamic 3D plotting of tensor data."""

    VALID_MODES: ClassVar[set] = set()
    fig: Optional[Any] = field(default=None, init=False)
    ax: Optional[Any] = field(default=None, init=False)

    def __post_init__(self):
        self.VALID_MODES = {mode.value for mode in PlotMode}

    def plot(
        self,
        tensor,
        mode: Literal["surface", "voxel", "scatter", "bar"] = "scatter",
        title="Tensor Visualizer",
    ):
        """Plot the tensor data in 3D."""
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mode: '{mode}'. Valid modes: {', '.join(self.VALID_MODES)}"
            )
        logging.info(f"Tensor shape: {tensor.shape}, Mode: {mode}")

        if hasattr(tensor, "cpu"):
            tensor = tensor.detach().cpu().numpy().astype(float)
        else:
            tensor = np.array(tensor, dtype=float)

        ndim = tensor.ndim
        if ndim not in (1, 2, 3):
            raise ValueError(
                f"Only 1D, 2D, or 3D tensors are supported, given: {ndim}D"
            )
        logging.info(f"Tensor shape: {tensor.shape}, Mode: {mode}")

        if ndim == 1:
            tensor = tensor[np.newaxis, np.newaxis, :]
        elif ndim == 2:
            tensor = tensor[np.newaxis, :, :]

        D, H, W = tensor.shape
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

        match mode:
            case "scatter":
                self._plot_scatter(tensor, D, H, W)

            case "surface":
                self._plot_surface(tensor, H, W)

            case "voxel":
                self._plot_voxel(tensor)

            case "bar":
                self._plot_bar(tensor)

        self.ax.set_xlabel("W (Width)")
        self.ax.set_ylabel("H (Height)")
        self.ax.set_zlabel("D (Depth)")
        self.ax.set_title(f"{title}\nShape: {tensor.shape} | Mode: {mode}")
        plt.tight_layout()
        plt.show()

    def _plot_scatter(self, tensor, D, H, W):
        d_idx, h_idx, w_idx = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing="ij"
        )

        x = w_idx.ravel()
        y = h_idx.ravel()
        z = d_idx.ravel()
        values = tensor.ravel()

        # norm = plt.Normalize(values.min(), values.max())
        # colors = cm.viridis(norm(values))

        sc = self.ax.scatter(x, y, z, c=values, cmap="viridis", s=40)
        plt.colorbar(sc, ax=self.ax, shrink=0.5, label="Values")

    def _plot_surface(self, tensor, H, W):
        slice_2d = tensor[0]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        surf = self.ax.plot_surface(
            X, Y, slice_2d, cmap="plasma", edgecolor="none", alpha=0.9
        )
        plt.colorbar(surf, ax=self.ax, shrink=0.5, label="Values")

    def _plot_voxel(self, tensor):
        norm_t = (tensor - tensor.min()) / (tensor.ptp() + 1e-9)
        colors_rgba = cm.coolwarm(norm_t)
        colors_rgba[..., 3] = 0.4 + 0.5 * norm_t
        self.ax.voxels(
            norm_t > 0.3, facecolors=colors_rgba, edgecolor="k", linewidth=0.3
        )

    def _plot_bar(self, tensor):
        slice_2d = tensor[0]
        _h, _w = slice_2d.shape
        xpos, ypos = np.meshgrid(np.arange(_w), np.arange(_h))
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        dz = slice_2d.ravel()

        norm = plt.Normalize(dz.min(), dz.max())
        colors = cm.magma(norm(dz))

        self.ax.bar3d(
            xpos,
            ypos,
            zpos,
            dx=0.8,
            dy=0.8,
            dz=dz,
            color=colors,
            alpha=0.85,
            shade=True,
        )

    def save(self, filename):
        """Save the current plot to a file."""
        if self.fig is None:
            raise RuntimeError("No plot to save. Please call plot() first.")
        self.fig.savefig(filename)
        logging.info(f"Plot saved to {filename}")


def to_image_3d(tensor, title="Tensor Visualizer", mode="scatter"):
    """Convert tensor data to 3D visualization."""
    ndim = tensor.ndim

    # Validate mode
    if mode not in ("scatter", "surface", "voxel", "bar"):
        raise ValueError(
            f"Unknown mode: '{mode}'. Valid modes: scatter, surface, voxel, bar"
        )
    logging.info(f"Tensor shape: {tensor.shape}, Mode: {mode}")

    # Check if tensor is 1D, 2D, or 3D
    if ndim not in (1, 2, 3):
        raise ValueError(f"Only 1D, 2D, or 3D tensors are supported, given: {ndim}D")
    logging.info(f"Tensor shape: {tensor.shape}, Mode: {mode}")

    if hasattr(tensor, "cpu"):
        tensor = tensor.detach().cpu().numpy().astype(float)
    else:
        tensor = np.array(tensor, dtype=float)

    match ndim:
        case 1:
            tensor = tensor[np.newaxis, np.newaxis, :]

        case 2:
            tensor = tensor[np.newaxis, :, :]

        case _ if ndim > 3:
            raise ValueError(f"Maximum 3D tensor supported, given: {ndim}D")

    D, H, W = tensor.shape
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    match mode:
        case "scatter":
            d_idx, h_idx, w_idx = np.meshgrid(
                np.arange(D), np.arange(H), np.arange(W), indexing="ij"
            )
            x = w_idx.ravel()
            y = h_idx.ravel()
            z = d_idx.ravel()
            values = tensor.ravel()

            norm = plt.Normalize(values.min(), values.max())
            colors = cm.viridis(norm(values))

            sc = ax.scatter(x, y, z, c=values, cmap="viridis", s=40)
            plt.colorbar(sc, ax=ax, shrink=0.5, label="Values")

        case "surface":
            slice_2d = tensor[0]
            X, Y = np.meshgrid(np.arange(W), np.arange(H))
            surf = ax.plot_surface(
                X, Y, slice_2d, cmap="plasma", edgecolor="none", alpha=0.9
            )
            plt.colorbar(surf, ax=ax, shrink=0.5, label="Values")

        case "voxel":
            norm_t = (tensor - tensor.min()) / (tensor.ptp() + 1e-9)
            colors_rgba = cm.coolwarm(norm_t)
            colors_rgba[..., 3] = 0.4 + 0.5 * norm_t
            ax.voxels(
                norm_t > 0.3, facecolors=colors_rgba, edgecolor="k", linewidth=0.3
            )

        case "bar":
            slice_2d = tensor[0]
            _h, _w = slice_2d.shape
            xpos, ypos = np.meshgrid(np.arange(_w), np.arange(_h))
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = np.zeros_like(xpos)
            dz = slice_2d.ravel()

            norm = plt.Normalize(dz.min(), dz.max())
            colors = cm.magma(norm(dz))

            ax.bar3d(
                xpos,
                ypos,
                zpos,
                dx=0.8,
                dy=0.8,
                dz=dz,
                color=colors,
                alpha=0.85,
                shade=True,
            )

    ax.set_xlabel("W (Width)")
    ax.set_ylabel("H (Height)")
    ax.set_zlabel("D (Depth)")
    ax.set_title(f"{title}\nShape: {tensor.shape} | Mode: {mode}")
    plt.tight_layout()
    plt.show()
    return fig, ax
