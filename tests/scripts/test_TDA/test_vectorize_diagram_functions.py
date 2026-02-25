import numpy as np
import pytest

from scripts.TDA.vectorize_diagram_functions import (
    vectorize_diagram,
    features_mid_lifetime,
    betti_curves_from_persistence,
    triangle_function,
    persistence_to_diagrams,
    persistence_landscape_1d,
)

def test_vectorize_diagram_empty():
    vec = vectorize_diagram([])

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.size > 0
    assert np.all(vec == 0)


def test_vectorize_diagram_basic():
    diagram = [
        (0, (0.0, 1.0)),
        (0, (0.2, 2.0)),
        (1, (0.5, 1.5)),
    ]

    vec = vectorize_diagram(diagram, n_bins=50)

    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.size > 0


def test_features_mid_lifetime():
    diagram = np.array([
        [0.0, 2.0],
        [1.0, 3.0],
    ])

    mid = features_mid_lifetime(diagram)

    assert np.allclose(mid, np.array([1.0, 2.0]))


def test_betti_curves_from_persistence():
    diagrams = [
        np.array([[0.0, 2.0], [1.0, 3.0]]),
        np.array([[0.5, 1.5]]),
    ]

    t, curves = betti_curves_from_persistence(diagrams, n_bins=10)

    assert len(t) == 10
    assert len(curves) == 2
    assert all(len(c) == 10 for c in curves)


def test_triangle_function_shape_and_values():
    eps = np.array([0.0, 0.5, 1.0, 1.5])
    birth = 0.0
    death = 1.0

    values = triangle_function(eps, birth, death)

    assert values.shape == eps.shape
    assert np.all(values >= 0)
    assert values[0] == 0
    assert values[-1] == 0


def test_persistence_to_diagrams():
    persistence = [
        (0, (0.0, 1.0)),
        (1, (0.5, 2.0)),
        (0, (1.0, 3.0)),
    ]

    diagrams = persistence_to_diagrams(persistence)

    assert len(diagrams) == 2
    assert diagrams[0].shape == (2, 2)
    assert diagrams[1].shape == (1, 2)


def test_persistence_landscape_empty():
    eps, landscape = persistence_landscape_1d(
        np.empty((0, 2)),
        n_layers=3,
        n_bins=10,
    )

    assert eps is None
    assert landscape.shape == (3, 10)
    assert np.all(landscape == 0)


def test_persistence_landscape_basic():
    diagram = np.array([
        [0.0, 2.0],
        [1.0, 3.0],
    ])

    eps, landscape = persistence_landscape_1d(
        diagram,
        n_layers=2,
        n_bins=20,
    )

    assert eps.shape == (20,)
    assert landscape.shape == (2, 20)
    assert np.all(landscape >= 0)