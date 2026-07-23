"""Unit + integration tests for GaussianProcess.save() and .load()."""

import pickle
from pathlib import Path

import numpy as np
import pytest
from gplite._utils._errors import ValidationError
from gplite.GaussianProcess.gaussian_process import GaussianProcess
from gplite.Kernels.rbf import RBFKernel


def _fitted_gp() -> GaussianProcess:
    x = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
    y = np.sin(x).ravel()
    gp = GaussianProcess(RBFKernel(length_scale=1.0))
    gp.fit(x, y)
    return gp


# ===========================================================================
# save()
# ===========================================================================


class TestGPSave:
    def test_save_creates_file(self, tmp_path: Path):
        gp = _fitted_gp()
        filepath = tmp_path / "model.pkl"
        gp.save(filepath)
        assert filepath.exists()

    def test_save_accepts_string_path(self, tmp_path: Path):
        gp = _fitted_gp()
        filepath = str(tmp_path / "model.pkl")
        gp.save(filepath)
        assert Path(filepath).exists()

    def test_save_invalid_path_type_raises(self):
        gp = _fitted_gp()
        with pytest.raises(ValidationError):
            gp.save(12345)


# ===========================================================================
# load()
# ===========================================================================


class TestGPLoad:
    def test_save_load_roundtrip(self, tmp_path: Path):
        gp = _fitted_gp()
        filepath = tmp_path / "model.pkl"
        gp.save(filepath)

        loaded = GaussianProcess.load(filepath)
        assert isinstance(loaded, GaussianProcess)

    def test_loaded_model_predictions_match(self, tmp_path: Path):
        gp = _fitted_gp()
        filepath = tmp_path / "model.pkl"
        gp.save(filepath)

        loaded = GaussianProcess.load(filepath)
        x_test = np.linspace(0, 2 * np.pi, 10).reshape(-1, 1)
        np.testing.assert_allclose(
            gp.predict(x_test), loaded.predict(x_test), atol=1e-10
        )

    def test_load_nonexistent_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            GaussianProcess.load(tmp_path / "nonexistent.pkl")

    def test_load_wrong_type_raises(self, tmp_path: Path):
        """Loading a file that doesn't contain a GaussianProcess should raise
        TypeError.
        """
        filepath = tmp_path / "bad.pkl"
        with filepath.open("wb") as f:
            pickle.dump({"not": "a gp"}, f)
        with pytest.raises(TypeError):
            GaussianProcess.load(filepath)

    def test_load_invalid_path_type_raises(self):
        with pytest.raises(ValidationError):
            GaussianProcess.load(12345)

    def test_load_unfitted_model_warns(self, tmp_path: Path):
        """Loading an unfitted model should emit a UserWarning."""
        gp = GaussianProcess(RBFKernel(length_scale=1.0))  # unfitted
        filepath = tmp_path / "unfitted.pkl"
        gp.save(filepath)

        with pytest.warns(UserWarning, match="not fitted"):
            loaded = GaussianProcess.load(filepath)
        assert loaded.alpha.size == 0

    def test_load_accepts_string_path(self, tmp_path: Path):
        gp = _fitted_gp()
        filepath = str(tmp_path / "model.pkl")
        gp.save(filepath)
        loaded = GaussianProcess.load(filepath)
        assert isinstance(loaded, GaussianProcess)
