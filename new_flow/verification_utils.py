# verification_utils.py
import sys
import numpy as np

def recursive_check(expected, actual, rtol=1e-3, atol=1e-5, path="output"):
    """
    Recursively compares two objects (handling numpy, torch, dicts, lists, primitives).
    Returns: (bool, str) -> (passed, error_message)
    """
    # 0. Pre-processing: Handle Torch Tensors (convert to numpy)
    # We check sys.modules to avoid hard dependency on torch if not installed
    if "torch" in sys.modules:
        import torch
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().cpu().numpy()

    # 1. Type Mismatch Check
    # Allow loose matching between tuple and list, otherwise strict
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        pass # Treat tuple and list as compatible sequences
    elif type(expected) != type(actual):
        # Special case: Scalar numpy array vs Python scalar
        if np.isscalar(expected) and isinstance(actual, np.ndarray) and actual.ndim == 0:
            actual = actual.item()
        elif np.isscalar(actual) and isinstance(expected, np.ndarray) and expected.ndim == 0:
            expected = expected.item()
        else:
            return False, f"Type mismatch at {path}: expected {type(expected)}, got {type(actual)}"

    # 2. Numpy Arrays
    if isinstance(expected, np.ndarray):
        if expected.shape != actual.shape:
            return False, f"Shape mismatch at {path}: expected {expected.shape}, got {actual.shape}"
        
        # Check for numeric types to use allclose
        if np.issubdtype(expected.dtype, np.number):
            if not np.allclose(expected, actual, rtol=rtol, atol=atol, equal_nan=True):
                diff = np.abs(expected - actual)
                max_diff = np.max(diff) if diff.size > 0 else 0
                return False, f"Value mismatch at {path}: max difference {max_diff} > tol"
        else:
            # String or Object arrays
            if not np.array_equal(expected, actual):
                return False, f"Non-numeric array mismatch at {path}"
        return True, ""

    # 3. Dictionaries
    if isinstance(expected, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False, f"Key mismatch at {path}: expected {list(expected.keys())}, got {list(actual.keys())}"
        for k in expected:
            res, msg = recursive_check(expected[k], actual[k], rtol, atol, path=f"{path}['{k}']")
            if not res: return False, msg
        return True, ""

    # 4. Lists / Tuples
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"Length mismatch at {path}: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            res, msg = recursive_check(e, a, rtol, atol, path=f"{path}[{i}]")
            if not res: return False, msg
        return True, ""

    # 5. Basic Types (float, int, str, bool, None)
    if isinstance(expected, float):
        if not np.isclose(expected, actual, rtol=rtol, atol=atol):
            return False, f"Float mismatch at {path}: expected {expected}, got {actual}"
    elif expected != actual:
        return False, f"Value mismatch at {path}: expected {expected}, got {actual}"

    return True, ""