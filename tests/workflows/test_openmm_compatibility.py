"""OpenMM compatibility tests for GaussianProcess.to_str().

These tests validate that every kernel's string representation can be:
  1. Accepted by OpenMM's expression parser (no parse errors).
  2. Evaluated inside an OpenMM simulation to a finite, non-NaN energy.
  3. Numerically consistent with GaussianProcess.predict() at the same point.

Strategy: Use CustomExternalForce, which exposes x, y, z as the particle
coordinates. This maps cleanly to 1-D (variable='x') and 2-D
(variables=['x', 'y']) GP strings without any per-parameter registration.
"""

import numpy as np
import pytest
from gplite.GaussianProcess.gaussian_process import (
    GaussianProcess,
)
from gplite.Kernels.constant import ConstantKernel
from gplite.Kernels.matern import MaternKernel
from gplite.Kernels.periodic import PeriodicKernel
from gplite.Kernels.rbf import RBFKernel

openmm = pytest.importorskip("openmm", reason="openmm not installed")


# mark every test in this module as slow
pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Shared OpenMM setup helpers
# ---------------------------------------------------------------------------

_PLATFORM = openmm.Platform.getPlatformByName("Reference")
_ATOL = 1e-4  # tolerance between real GP prediction and string function


def _build_context_1d(expression: str, x_val: float) -> openmm.Context:
    """Build a single-particle OpenMM context using the expression as a
    CustomExternalForce energy function evaluated at (x_val, 0, 0).
    """
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.CustomExternalForce(expression)
    force.addParticle(0, [])
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integrator, _PLATFORM)
    ctx.setPositions([[x_val, 0.0, 0.0]])
    return ctx


def _build_context_2d(expression: str, x_val: float, y_val: float) -> openmm.Context:
    """Build a single-particle context with the 2-D expression evaluated
    at (x_val, y_val, 0).
    """
    system = openmm.System()
    system.addParticle(1.0)
    force = openmm.CustomExternalForce(expression)
    force.addParticle(0, [])
    system.addForce(force)
    integrator = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(system, integrator, _PLATFORM)
    ctx.setPositions([[x_val, y_val, 0.0]])
    return ctx


def _energy(ctx: openmm.Context) -> float:
    """Return potential energy in kJ/mol as a plain float."""
    state = ctx.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(openmm.unit.kilojoules_per_mole)


def _fit_gp_1d(kernel, n: int = 12) -> GaussianProcess:
    x = np.linspace(0, np.pi, n).reshape(-1, 1)
    y = np.sin(x).ravel()
    gp = GaussianProcess(kernel)
    gp.fit(x, y)
    return gp


def _fit_gp_2d(kernel, n: int = 15) -> GaussianProcess:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.5, 1.5, size=(n, 2))
    y = x[:, 0] ** 2 + x[:, 1] ** 2
    gp = GaussianProcess(kernel)
    gp.fit(x, y)
    return gp


# ===========================================================================
# RBF kernel - isotropic (1D) and anisotropic (2D)
# ===========================================================================


class TestOpenMMRBFKernel:
    def test_isotropic_1d_parses(self):
        gp = _fit_gp_1d(RBFKernel(length_scale=1.0))
        expr = gp.to_str("x")
        # will raise if OpenMM cannot parse the expression
        ctx = _build_context_1d(expr, x_val=1.0)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_isotropic_1d_matches_predict(self):
        gp = _fit_gp_1d(RBFKernel(length_scale=1.0))
        expr = gp.to_str("x")
        x_eval = 1.2
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"RBF iso 1D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )

    def test_anisotropic_2d_parses(self):
        gp = _fit_gp_2d(RBFKernel(length_scale=[1.0, 2.0], isotropic=False))
        expr = gp.to_str(["x", "y"])
        ctx = _build_context_2d(expr, x_val=0.5, y_val=-0.5)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_anisotropic_2d_matches_predict(self):
        gp = _fit_gp_2d(RBFKernel(length_scale=[1.0, 2.0], isotropic=False))
        expr = gp.to_str(["x", "y"])
        x_eval, y_eval = 0.5, -0.5
        gp_val = float(gp.predict(np.array([[x_eval, y_eval]]))[0])
        ctx = _build_context_2d(expr, x_val=x_eval, y_val=y_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"RBF aniso 2D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )


# ===========================================================================
# Matern kernel - both nu values, isotropic and anisotropic
# ===========================================================================


class TestOpenMMMaternKernel:
    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_isotropic_1d_parses(self, nu):
        gp = _fit_gp_1d(MaternKernel(length_scale=1.0, nu=nu))
        expr = gp.to_str("x")
        ctx = _build_context_1d(expr, x_val=1.0)
        e = _energy(ctx)
        assert np.isfinite(e)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_isotropic_1d_matches_predict(self, nu):
        gp = _fit_gp_1d(MaternKernel(length_scale=1.0, nu=nu))
        expr = gp.to_str("x")
        x_eval = 1.2
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"Matern nu={nu} iso 1D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_anisotropic_2d_parses(self, nu):
        gp = _fit_gp_2d(MaternKernel(length_scale=[1.0, 1.5], nu=nu, isotropic=False))
        expr = gp.to_str(["x", "y"])
        ctx = _build_context_2d(expr, x_val=0.5, y_val=-0.3)
        e = _energy(ctx)
        assert np.isfinite(e)

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    def test_anisotropic_2d_matches_predict(self, nu):
        gp = _fit_gp_2d(MaternKernel(length_scale=[1.0, 1.5], nu=nu, isotropic=False))
        expr = gp.to_str(["x", "y"])
        x_eval, y_eval = 0.5, -0.3
        gp_val = float(gp.predict(np.array([[x_eval, y_eval]]))[0])
        ctx = _build_context_2d(expr, x_val=x_eval, y_val=y_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"Matern nu={nu} aniso 2D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )


# ===========================================================================
# Periodic kernel - isotropic and anisotropic
# ===========================================================================


class TestOpenMMPeriodicKernel:
    def test_isotropic_1d_parses(self):
        gp = _fit_gp_1d(PeriodicKernel(length_scale=1.0, period=np.pi))
        expr = gp.to_str("x")
        ctx = _build_context_1d(expr, x_val=1.0)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_isotropic_1d_matches_predict(self):
        gp = _fit_gp_1d(PeriodicKernel(length_scale=1.0, period=np.pi))
        expr = gp.to_str("x")
        x_eval = 1.2
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"Periodic iso 1D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )

    def test_anisotropic_2d_parses(self):
        gp = _fit_gp_2d(
            PeriodicKernel(
                length_scale=[1.0, 1.5],
                period=[np.pi, 2 * np.pi],
                isotropic=False,
            )
        )
        expr = gp.to_str(["x", "y"])
        ctx = _build_context_2d(expr, x_val=0.5, y_val=-0.3)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_anisotropic_2d_matches_predict(self):
        gp = _fit_gp_2d(
            PeriodicKernel(
                length_scale=[1.0, 1.5],
                period=[np.pi, 2 * np.pi],
                isotropic=False,
            )
        )
        expr = gp.to_str(["x", "y"])
        x_eval, y_eval = 0.5, -0.3
        gp_val = float(gp.predict(np.array([[x_eval, y_eval]]))[0])
        ctx = _build_context_2d(expr, x_val=x_eval, y_val=y_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"Periodic aniso 2D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )


# ===========================================================================
# Constant kernel
# ===========================================================================


class TestOpenMMConstantKernel:
    def test_1d_parses(self):
        gp = _fit_gp_1d(ConstantKernel(constant=2.0))
        expr = gp.to_str("x")
        ctx = _build_context_1d(expr, x_val=1.0)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_1d_matches_predict(self):
        gp = _fit_gp_1d(ConstantKernel(constant=2.0))
        expr = gp.to_str("x")
        x_eval = 1.2
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"Constant 1D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )


# ===========================================================================
# Composite kernels - additive and product
# ===========================================================================


class TestOpenMMCompositeKernels:
    def test_additive_isotropic_1d_parses(self):
        kernel = RBFKernel(length_scale=1.0) + ConstantKernel(constant=1.0)
        gp = _fit_gp_1d(kernel)
        expr = gp.to_str("x")
        ctx = _build_context_1d(expr, x_val=1.0)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_additive_isotropic_1d_matches_predict(self):
        kernel = RBFKernel(length_scale=1.0) + ConstantKernel(constant=1.0)
        gp = _fit_gp_1d(kernel)
        expr = gp.to_str("x")
        x_eval = 1.2
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"RBF+Constant iso 1D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )

    def test_product_isotropic_1d_parses(self):
        kernel = ConstantKernel(constant=2.0) * RBFKernel(length_scale=1.0)
        gp = _fit_gp_1d(kernel)
        expr = gp.to_str("x")
        ctx = _build_context_1d(expr, x_val=1.0)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_product_isotropic_1d_matches_predict(self):
        kernel = ConstantKernel(constant=2.0) * RBFKernel(length_scale=1.0)
        gp = _fit_gp_1d(kernel)
        expr = gp.to_str("x")
        x_eval = 1.2
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"Constant*RBF iso 1D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )

    def test_additive_anisotropic_2d_parses(self):
        kernel = RBFKernel(length_scale=[1.0, 2.0], isotropic=False) + MaternKernel(
            length_scale=[0.5, 1.5], nu=2.5, isotropic=False
        )
        gp = _fit_gp_2d(kernel)
        expr = gp.to_str(["x", "y"])
        ctx = _build_context_2d(expr, x_val=0.5, y_val=-0.5)
        e = _energy(ctx)
        assert np.isfinite(e)

    def test_additive_anisotropic_2d_matches_predict(self):
        kernel = RBFKernel(length_scale=[1.0, 2.0], isotropic=False) + MaternKernel(
            length_scale=[0.5, 1.5], nu=2.5, isotropic=False
        )
        gp = _fit_gp_2d(kernel)
        expr = gp.to_str(["x", "y"])
        x_eval, y_eval = 0.5, -0.5
        gp_val = float(gp.predict(np.array([[x_eval, y_eval]]))[0])
        ctx = _build_context_2d(expr, x_val=x_eval, y_val=y_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"RBF+Matern aniso 2D: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )

    def test_rbf_plus_matern_iso_1d_parses(self):
        kernel = RBFKernel(length_scale=1.0) + MaternKernel(length_scale=1.0, nu=1.5)
        gp = _fit_gp_1d(kernel)
        expr = gp.to_str("x")
        ctx = _build_context_1d(expr, x_val=1.0)
        assert np.isfinite(_energy(ctx))

    def test_periodic_plus_rbf_iso_1d_parses(self):
        kernel = PeriodicKernel(length_scale=1.0, period=np.pi) + RBFKernel(
            length_scale=1.0
        )
        gp = _fit_gp_1d(kernel)
        expr = gp.to_str("x")
        ctx = _build_context_1d(expr, x_val=1.0)
        assert np.isfinite(_energy(ctx))


# ===========================================================================
# Numerical consistency across multiple evaluation points
# ===========================================================================


class TestOpenMMNumericalConsistency:
    """Check GP predict vs OpenMM energy at several x values for robustness."""

    @pytest.mark.parametrize("x_eval", [0.0, 0.5, 1.0, 1.5, 2.0, np.pi])
    def test_rbf_iso_multipoint(self, x_eval):
        gp = _fit_gp_1d(RBFKernel(length_scale=1.0))
        expr = gp.to_str("x")
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"x={x_eval}: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )

    @pytest.mark.parametrize("nu", [1.5, 2.5])
    @pytest.mark.parametrize("x_eval", [0.3, 1.0, 2.5])
    def test_matern_iso_multipoint(self, nu, x_eval):
        gp = _fit_gp_1d(MaternKernel(length_scale=1.0, nu=nu))
        expr = gp.to_str("x")
        gp_val = float(gp.predict(np.array([[x_eval]]))[0])
        ctx = _build_context_1d(expr, x_val=x_eval)
        openmm_val = _energy(ctx)
        assert abs(gp_val - openmm_val) < _ATOL, (
            f"nu={nu} x={x_eval}: GP={gp_val:.6f}, OpenMM={openmm_val:.6f}"
        )
