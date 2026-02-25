import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from scripts.TDA.create_persistence_diagram_functions import delete_noise_from_diag, create_persistence_diagram, compute_pd_for_multichannel_image

def test_delete_noise_from_diag_filters_by_lifetime():
    persistence = [
        (0, (0.0, 1.0)),   # lifetime = 1
        (0, (0.0, 2.0)),   # lifetime = 2
        (1, (0.5, 3.0)),   # lifetime = 2.5
    ]

    result = delete_noise_from_diag(persistence, quantile=0.5)

    # остаются элементы с lifetime >= медианы
    assert len(result) == 2
    assert (0, (0.0, 2.0)) in result
    assert (1, (0.5, 3.0)) in result


def test_delete_noise_from_diag_empty():
    assert delete_noise_from_diag([]) == []


@patch("scripts.TDA.create_persistence_diagram_functions.gd.CubicalComplex")
def test_create_persistence_diagram_basic(mock_cc):
    mock_instance = mock_cc.return_value

    mock_instance.persistence.return_value = [
        (0, (0.0, 1.0)),
        (1, (0.2, 2.0)),
        (2, (0.1, 3.0)),  # должен быть отброшен (dim > max_dim)
        (0, (0.5, np.inf)),  # inf - отброшен
    ]

    img = np.random.randint(0, 255, (5, 5), dtype=np.uint8)

    result = create_persistence_diagram(img, max_dim=1, noise_quantile=0.0)

    assert isinstance(result, list)
    assert all(dim <= 1 for dim, _ in result)
    assert all(not np.isinf(death) for _, (_, death) in result)


@patch("scripts.TDA.create_persistence_diagram_functions.gd.CubicalComplex")
def test_create_persistence_diagram_basic(mock_cc):
    mock_instance = mock_cc.return_value

    mock_instance.persistence.return_value = [
        (0, (0.0, 1.0)),
        (1, (0.2, 2.0)),
        (2, (0.1, 3.0)),  # должен быть отброшен (dim > max_dim)
        (0, (0.5, np.inf)),  # inf - отброшен
    ]

    img = np.random.randint(0, 255, (5, 5), dtype=np.uint8)

    result = create_persistence_diagram(img, max_dim=1, noise_quantile=0.0)

    assert isinstance(result, list)
    assert all(dim <= 1 for dim, _ in result)
    assert all(not np.isinf(death) for _, (_, death) in result)