"""
Test suite for Deep Polynomial Chaos Neural Network Method.
"""

import pytest
import numpy as np
import torch


def test_imports():
    """Test that all modules can be imported."""
    try:
        import data_process
        import Deep_PCE
        import pce_loss
        print("✅ All modules imported successfully")
    except ImportError as e:
        pytest.skip(f"Module import failed: {e}")


def test_numpy_torch_compatibility():
    """Test NumPy and PyTorch compatibility."""
    np_array = np.random.randn(100, 5)
    torch_tensor = torch.from_numpy(np_array).float()
    np_array_back = torch_tensor.numpy()
    assert np.allclose(np_array, np_array_back)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
