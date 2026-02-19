import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def create_persistence_diagram(img_gray):
    """Создание диаграммы персистентности"""
    img_float = img_gray.astype(float)

    # строим кубический комплекс
    cubical_complex = gd.CubicalComplex(
      dimensions=img_float.shape,
      top_dimensional_cells=img_float.flatten()
    )

    # считаем персистентную гомологию
    cubical_complex.compute_persistence()

    persistence = cubical_complex.persistence()

    filtered_persistence = [
      (dim, (bith, death))
      for dim, (bith, death) in persistence
      if not np.isinf(death)
    ]

    return filtered_persistence

def delete_noise_from_diag(persistence, quantile=0.5):
    """Удаление шумовых точек с диаграммы персистентности"""
    births_deaths = [death-birth for dim, (birth, death) in persistence]
    births_deaths_array = np.array(births_deaths)
    quantile_lifetime = np.quantile(births_deaths_array, quantile)

    filtered_pairs = [(dim, (birth, death))
    for dim, (birth, death) in persistence
    if death-birth >= quantile_lifetime]

    return filtered_pairs

