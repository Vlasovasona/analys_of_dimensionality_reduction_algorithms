import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import gudhi as gd
from typing import List, Tuple, Dict

def create_persistence_diagram(
    img: np.ndarray,
    max_dim: int = 1,
    noise_quantile: float = 0.90
) -> List[Tuple[int, Tuple[float, float]]]:
    """
    Строит очищенную диаграмму персистентности для 2D изображения.
    """

    img = img.astype(np.float32)

    cc = gd.CubicalComplex(
        dimensions=img.shape,
        top_dimensional_cells=img.flatten()
    )

    cc.compute_persistence()

    persistence = [
        (dim, (birth, death))
        for dim, (birth, death) in cc.persistence()
        if dim <= max_dim and not np.isinf(death)
    ]

    return delete_noise_from_diag(persistence, quantile=noise_quantile)


def compute_pd_for_multichannel_image(
    img_3ch: np.ndarray,
    max_dim: int = 1,
    noise_quantile: float = 0.90
) -> Dict[str, list]:
    """
    Считает очищенные PD для каждого канала изображения.
    """
    return {
        "gray": create_persistence_diagram(
            img_3ch[:, :, 0], max_dim, noise_quantile
        ),
        "sobel": create_persistence_diagram(
            img_3ch[:, :, 1], max_dim, noise_quantile
        ),
        "gaussian": create_persistence_diagram(
            img_3ch[:, :, 2], max_dim, noise_quantile
        ),
    }


def delete_noise_from_diag(
    persistence: List[Tuple[int, Tuple[float, float]]],
    quantile: float = 0.90
) -> List[Tuple[int, Tuple[float, float]]]:
    """
    Удаляет шумовые точки из диаграммы персистентности
    на основе времени жизни (death - birth).

    Parameters
    ----------
    persistence : list of (dim, (birth, death))
        Диаграмма персистентности.
    quantile : float
        Квантиль для отсечения шумовых точек.

    Returns
    -------
    list of (dim, (birth, death))
        Очищенная диаграмма персистентности.
    """
    if not persistence:
        return persistence

    lifetimes = np.array([death - birth for _, (birth, death) in persistence])

    threshold = np.quantile(lifetimes, quantile)

    return [
        (dim, (birth, death))
        for dim, (birth, death) in persistence
        if (death - birth) >= threshold
    ]

