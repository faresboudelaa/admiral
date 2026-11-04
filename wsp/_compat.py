"""Import shims so PyQuaticus runs on machines without its optional robot stack."""

import sys
import types


def ensure_pymoos():
    """PyQuaticus' base_policies import the MOOS bridge, which needs `pymoos`.

    pymoos is a C++ extension that does not build on Windows without MSVC, and
    nothing in simulation uses it, so an empty stand-in module is enough.
    """
    try:
        import pymoos  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pymoos")
        stub.comms = type("comms", (), {})
        sys.modules["pymoos"] = stub
