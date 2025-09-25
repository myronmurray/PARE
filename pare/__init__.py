from collections import namedtuple
import inspect
import numpy as np

if not hasattr(inspect, "getargspec"):
    ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

    def _getargspec(func):
        full = inspect.getfullargspec(func)
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)

    inspect.getargspec = _getargspec  # type: ignore[attr-defined]

# Backwards compatibility for chumpy with newer NumPy
# NumPy 2.0 removed legacy aliases such as np.bool; avoid attribute access to
# prevent FutureWarning on NumPy 1.26.x.
if "bool" not in np.__dict__:
    np.bool = bool  # type: ignore[attr-defined]
if "int" not in np.__dict__:
    np.int = int  # type: ignore[attr-defined]
if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]
if "complex" not in np.__dict__:
    np.complex = complex  # type: ignore[attr-defined]
if "object" not in np.__dict__:
    np.object = object  # type: ignore[attr-defined]
if "unicode" not in np.__dict__:
    np.unicode = str  # type: ignore[attr-defined]
if "str" not in np.__dict__:
    np.str = str  # type: ignore[attr-defined]
