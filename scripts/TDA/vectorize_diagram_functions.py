import numpy as np
from collections import defaultdict


def features_lifetime(persistence):
    """Время жизни признаков"""
    births_deaths = [death-birth for dim, (birth, death) in persistence]
    births_deaths_array = np.array(births_deaths)

    return births_deaths_array


def features_mid_lifetime(persistence):
    """Середина жизни признаков"""
    births_deaths = [(death+birth)/2 for dim, (birth, death) in persistence]
    mid_births_deaths_array = np.array(births_deaths)

    return mid_births_deaths_array


def persistence_to_diagrams(filtered_persistence):
    """
    -> diagrams[k] = np.array([[birth, death], ...]) для H_k
    """
    diagrams = defaultdict(list)

    for dim, (birth, death) in filtered_persistence:
        diagrams[dim].append([birth, death])

    # преобразуем в список numpy-массивов
    max_dim = max(diagrams.keys())
    return [
        np.array(diagrams[k]) if k in diagrams else np.empty((0, 2))
        for k in range(max_dim + 1)
    ]


def betti_curves_from_persistence(diagrams, n_bins=200):
    t_min = min(
        np.min(d[:, 0]) for d in diagrams if len(d) > 0
    )
    t_max = max(
        np.max(d[:, 1]) for d in diagrams if len(d) > 0
    )

    t = np.linspace(t_min, t_max, n_bins)
    betti_curves = []

    for dgm in diagrams:
        curve = np.zeros_like(t)
        for birth, death in dgm:
            curve += (t >= birth) & (t < death)
        betti_curves.append(curve)

    return t, betti_curves


def triangle_function(eps, birth, death):
    return np.maximum(
        0.0,
        np.minimum(eps - birth, death - eps)
    )


def persistence_landscape_1d(diagram, n_layers=3, n_bins=200, eps_min=None, eps_max=None):
    """
    diagram: np.array of shape (n_intervals, 2)
    returns:
        eps: ось ε
        landscape: shape (n_layers, n_bins)
    """
    if len(diagram) == 0:
        return None, np.zeros((n_layers, n_bins))

    if eps_min is None:
        eps_min = np.min(diagram[:, 0])
    if eps_max is None:
        eps_max = np.max(diagram[:, 1])

    eps = np.linspace(eps_min, eps_max, n_bins)

    # значения Λ_t(ε) для всех t
    values = np.zeros((len(diagram), n_bins))

    for i, (b, d) in enumerate(diagram):
        values[i] = triangle_function(eps, b, d)

    # сортировка по убыванию → max_k
    values_sorted = np.sort(values, axis=0)[::-1]

    landscape = values_sorted[:n_layers]

    return eps, landscape