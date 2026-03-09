import numpy as np
from collections import defaultdict
from scipy.stats import skew, kurtosis


def vectorize_diagram(
    diagram,
    n_bins: int = 50,
    n_layers: int = 2,
):
    MAX_DIM = 2

    N_STATS = 12
    N_MID = 5
    LANDSCAPE_STATS_DIM = 5  # max, mean, std, sum, norm

    VECTOR_SIZE = (
        N_STATS
        + N_MID
        + (MAX_DIM + 1) * LANDSCAPE_STATS_DIM
    )

    # Пустая диаграмма
    if len(diagram) == 0:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

    # Lifetimes
    lifetimes = np.array(
        [death - birth for _, (birth, death) in diagram],
        dtype=np.float32,
    )

    # Статистика lifetimes
    lifetime_stats = np.array([
        len(lifetimes),
        np.max(lifetimes),
        np.sum(lifetimes),
        np.mean(lifetimes),
        np.std(lifetimes),
        np.sum(np.abs(lifetimes)),
        np.linalg.norm(lifetimes),
        np.quantile(lifetimes, 0.25),
        np.quantile(lifetimes, 0.5),
        np.quantile(lifetimes, 0.75),
        kurtosis(lifetimes, fisher=True, bias=False),
        skew(lifetimes, bias=False),
    ], dtype=np.float32)

    # Mid-life
    mid_life = np.quantile(
        lifetimes / 2, [0, 0.25, 0.5, 0.75, 1]
    ).astype(np.float32)

    # Persistence landscapes → статистики
    diagrams = persistence_to_diagrams(diagram)
    landscape_features = []

    for k in range(MAX_DIM + 1):
        if k < len(diagrams):
            _, landscape = persistence_landscape_1d(
                diagrams[k],
                n_layers=n_layers,
                n_bins=n_bins,
            )
        else:
            landscape = np.zeros((n_layers, n_bins))

        landscape_features.append(landscape_stats(landscape))

    landscape_vector = np.concatenate(landscape_features).astype(np.float32)

    return np.concatenate([
        lifetime_stats,
        mid_life,
        landscape_vector,
    ])


def features_mid_lifetime(diagram):
    """Середина жизни признаков"""
    mid = (diagram[:, 0] + diagram[:, 1]) / 2
    return mid


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

def persistence_landscape_1d(diagram, n_layers=3, n_bins=200, eps_min=None, eps_max=None):
    """
    diagram: np.array (n_intervals, 2)
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

    values = np.zeros((len(diagram), n_bins))

    for i, (b, d) in enumerate(diagram):
        values[i] = triangle_function(eps, b, d)

    # сортировка по убыванию → max_k
    values_sorted = np.sort(values, axis=0)[::-1]

    landscape = values_sorted[:n_layers]

    return eps, landscape

def landscape_stats(landscape):
    return np.array([
        np.max(landscape),
        np.mean(landscape),
        np.std(landscape),
        np.sum(landscape),
        np.linalg.norm(landscape),
    ], dtype=np.float32)