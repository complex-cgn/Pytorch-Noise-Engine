import logging
import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.cm as cm

def to_image_3d(tensor, title="Tensor 3D Görselleştirme", mode="scatter"):
    """
    Girdi tensör verisini 3D olarak görselleştirir.

    Parametreler
    ----------
    tensor : array-like
        Görselleştirilecek tensör. 1D, 2D veya 3D numpy dizisi kabul eder.
        - 1D  → z ekseninde bar grafik
        - 2D  → yüzey (surface) grafiği
        - 3D  → scatter veya voxel grafiği

    title : str
        Grafik başlığı.

    mode : str
        "scatter"  → 3D nokta bulutu (tüm boyutlar için)
        "surface"  → yüzey grafiği (yalnızca 2D tensör için iyi çalışır)
        "voxel"    → voxel (küp) grafiği (3D tensör için)
        "bar"      → çubuk grafik (1D/2D tensörler için)

    Döndürür
    -------
    fig, ax : matplotlib Figure ve Axes3D nesneleri
    """
    if hasattr(tensor, 'cpu'):
        tensor = tensor.detach().cpu().numpy().astype(float)
    else:
        tensor = np.array(tensor, dtype=float)
    ndim = tensor.ndim

    # Tensörü 3D'ye yükselт
    if ndim == 1:
        tensor = tensor[np.newaxis, np.newaxis, :]   # (1, 1, N)
    elif ndim == 2:
        tensor = tensor[np.newaxis, :, :]            # (1, H, W)
    elif ndim > 3:
        raise ValueError(f"Maksimum 3D tensör desteklenir, verilen: {ndim}D")

    D, H, W = tensor.shape
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # ── SCATTER ──────────────────────────────────────────────────────────────
    if mode == "scatter":
        d_idx, h_idx, w_idx = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing="ij"
        )
        x = w_idx.ravel()
        y = h_idx.ravel()
        z = d_idx.ravel()
        values = tensor.ravel()

        norm = plt.Normalize(values.min(), values.max())
        colors = cm.viridis(norm(values))

        sc = ax.scatter(x, y, z, c=values, cmap="viridis",
                        s=40, alpha=0.8, edgecolors="none")
        plt.colorbar(sc, ax=ax, shrink=0.5, label="Değer")

    # ── SURFACE ──────────────────────────────────────────────────────────────
    elif mode == "surface":
        slice_2d = tensor[0]  # ilk (veya tek) derinlik katmanı
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        surf = ax.plot_surface(X, Y, slice_2d,
                               cmap="plasma", edgecolor="none", alpha=0.9)
        plt.colorbar(surf, ax=ax, shrink=0.5, label="Değer")

    # ── VOXEL ─────────────────────────────────────────────────────────────────
    elif mode == "voxel":
        norm_t = (tensor - tensor.min()) / (tensor.ptp() + 1e-9)
        colors_rgba = cm.coolwarm(norm_t)
        colors_rgba[..., 3] = 0.4 + 0.5 * norm_t  # alfa: düşük değer şeffaf
        ax.voxels(norm_t > 0.3, facecolors=colors_rgba, edgecolor="k",
                  linewidth=0.3)

    # ── BAR ───────────────────────────────────────────────────────────────────
    elif mode == "bar":
        slice_2d = tensor[0]
        _h, _w = slice_2d.shape
        xpos, ypos = np.meshgrid(np.arange(_w), np.arange(_h))
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros_like(xpos)
        dz = slice_2d.ravel()

        norm = plt.Normalize(dz.min(), dz.max())
        colors = cm.magma(norm(dz))

        ax.bar3d(xpos, ypos, zpos,
                 dx=0.8, dy=0.8, dz=dz,
                 color=colors, alpha=0.85, shade=True)

    else:
        raise ValueError(f"Bilinmeyen mod: '{mode}'. "
                         "Geçerli modlar: scatter, surface, voxel, bar")

    ax.set_xlabel("W (Genişlik)")
    ax.set_ylabel("H (Yükseklik)")
    ax.set_zlabel("D (Derinlik)")
    ax.set_title(f"{title}\nŞekil: {tensor.shape} | Mod: {mode}")
    plt.tight_layout()
    plt.show()
    return fig, ax

def to_image_1D(
    image_tensor: torch.Tensor,
    plot_title: str,
    output_path: str,
    color_map: str,
    dpi: int,
    show_plot: bool,
):
    """
    Tensor-to-image conversion utility for 1D tensors.

    Supports saving 1D tensors as images using matplotlib, with optional
    color mapping and plot display.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info("Output folder can't found creating new one...")
        os.makedirs(output_dir)

    # Save as image
    plt.figure(figsize=(2, 2))
    plt.plot(image_tensor.cpu().numpy(), color=color_map)
    plt.axis("off")
    plt.title(plot_title, fontname="DejaVu Sans", fontsize=16)
    plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    plt.close()


def to_image_2D(
    image_tensor: torch.Tensor,
    plot_title: str,
    output_path: str,
    color_map: str,
    dpi: int,
    show_plot: bool,
):
    """
    Tensor-to-image conversion utility.

    Supports saving tensors as images using matplotlib, with optional
    color mapping and plot display.
    """

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        logging.info("Output folder can't found creating new one...")
        os.makedirs(output_dir)

    # Save as image
    plt.figure(figsize=(2, 2))
    plt.imshow(image_tensor.cpu().numpy(), cmap=color_map, origin="upper")
    plt.axis("off")
    plt.title(plot_title, fontname="DejaVu Sans", fontsize=16)
    plt.savefig(output_path, dpi=dpi)

    if show_plot:
        plt.show()
    plt.close()

    logging.info(f"Noise generated and saved to {output_path}")

