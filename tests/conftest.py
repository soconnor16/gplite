"""Shared pytest fixtures for the gplite test suite.

All test modules can import these fixtures by simply declaring them as
function parameters - pytest discovers them automatically.
"""

import numpy as np
import pytest
from gplite.ActiveLearning.active_learning import ActiveLearner
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels.constant import ConstantKernel
from gplite.Kernels.matern import MaternKernel
from gplite.Kernels.periodic import PeriodicKernel
from gplite.Kernels.rbf import RBFKernel

# ---------------------------------------------------------------------------
# Small, deterministic datasets
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random number generator for reproducible tests."""
    return np.random.default_rng(0)


@pytest.fixture
def x_1d() -> np.ndarray:
    """30 evenly spaced points in [0, 2π], 1-D input."""
    return np.linspace(0, 2 * np.pi, 30).reshape(-1, 1)


@pytest.fixture
def y_1d(x_1d: np.ndarray) -> np.ndarray:
    """Noiseless sine wave targets for x_1d."""
    return np.sin(x_1d).ravel()


@pytest.fixture
def x_2d(rng: np.random.Generator) -> np.ndarray:
    """40 random 2-D points in [-2, 2]²."""
    return rng.uniform(-2, 2, size=(40, 2))


@pytest.fixture
def y_2d(x_2d: np.ndarray) -> np.ndarray:
    """Simple quadratic surface: x₀² + x₁²."""
    return x_2d[:, 0] ** 2 + x_2d[:, 1] ** 2


# ---------------------------------------------------------------------------
# Isotropic kernel instances
# ---------------------------------------------------------------------------


@pytest.fixture
def rbf_iso() -> RBFKernel:
    """Isotropic RBF kernel with length_scale=1.0."""
    return RBFKernel(length_scale=1.0, isotropic=True)


@pytest.fixture
def matern_iso_15() -> MaternKernel:
    """Isotropic Matérn ν=1.5 kernel with length_scale=1.0."""
    return MaternKernel(length_scale=1.0, nu=1.5, isotropic=True)


@pytest.fixture
def matern_iso_25() -> MaternKernel:
    """Isotropic Matérn ν=2.5 kernel with length_scale=1.0."""
    return MaternKernel(length_scale=1.0, nu=2.5, isotropic=True)


@pytest.fixture
def periodic_iso() -> PeriodicKernel:
    """Isotropic Periodic kernel with length_scale=1.0, period=2.0."""
    return PeriodicKernel(length_scale=1.0, period=2.0, isotropic=True)


@pytest.fixture
def constant_kernel() -> ConstantKernel:
    """Constant kernel with constant=2.0."""
    return ConstantKernel(constant=2.0)


# ---------------------------------------------------------------------------
# Anisotropic kernel instances (2-D)
# ---------------------------------------------------------------------------


@pytest.fixture
def rbf_aniso() -> RBFKernel:
    """Anisotropic RBF kernel with per-dimension length scales."""
    return RBFKernel(length_scale=[1.0, 2.0], isotropic=False)


@pytest.fixture
def matern_aniso_15() -> MaternKernel:
    """Anisotropic Matérn ν=1.5 kernel."""
    return MaternKernel(length_scale=[1.0, 2.0], nu=1.5, isotropic=False)


@pytest.fixture
def matern_aniso_25() -> MaternKernel:
    """Anisotropic Matérn ν=2.5 kernel."""
    return MaternKernel(length_scale=[1.0, 2.0], nu=2.5, isotropic=False)


@pytest.fixture
def periodic_aniso() -> PeriodicKernel:
    """Anisotropic Periodic kernel."""
    return PeriodicKernel(
        length_scale=[1.0, 1.5],
        period=[2.0, 3.0],
        isotropic=False,
    )


# ---------------------------------------------------------------------------
# Fitted GP instances
# ---------------------------------------------------------------------------


@pytest.fixture
def fitted_gp_1d(x_1d: np.ndarray, y_1d: np.ndarray) -> GaussianProcess:
    """GP fitted on the 1-D sine dataset (no optimization, fast)."""
    gp = GaussianProcess(RBFKernel(length_scale=1.0))
    gp.fit(x_1d, y_1d)
    return gp


@pytest.fixture
def fitted_gp_2d(x_2d: np.ndarray, y_2d: np.ndarray) -> GaussianProcess:
    """GP fitted on the 2-D quadratic dataset (no optimization, fast)."""
    gp = GaussianProcess(RBFKernel(length_scale=1.0))
    gp.fit(x_2d, y_2d)
    return gp


# ---------------------------------------------------------------------------
# Active learner
# ---------------------------------------------------------------------------


@pytest.fixture
def al_dataset() -> tuple[np.ndarray, np.ndarray]:
    """50-point sine dataset for active learning tests."""
    x = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
    y = np.sin(x).ravel()
    return x, y


@pytest.fixture
def active_learner(al_dataset: tuple[np.ndarray, np.ndarray]) -> ActiveLearner:
    """An ActiveLearner initialized on the 50-point sine dataset."""
    x, y = al_dataset
    return ActiveLearner(kernel=RBFKernel(length_scale=1.0), x_full=x, y_full=y)
