"""Top-level MAOL training entrypoint.

The NumPy backend was removed during the MAOIDL -> MAOL reconstruction.  The
``--backend torch`` spelling is still accepted for old shell aliases; any other
backend fails with a clear message.
"""

from __future__ import annotations

import sys


def _pop_backend_arg(argv: list[str], default: str = "torch") -> tuple[str, list[str]]:
    backend = default
    out: list[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--backend":
            if index + 1 >= len(argv):
                raise SystemExit("--backend requires a value (torch)")
            backend = argv[index + 1].strip().lower()
            skip_next = True
            continue
        if arg.startswith("--backend="):
            backend = arg.split("=", 1)[1].strip().lower()
            continue
        out.append(arg)
    if backend != "torch":
        raise SystemExit(f"Unsupported backend: {backend}. NumPy support was removed; use --backend torch.")
    return backend, out


def main() -> None:
    _, forwarded_argv = _pop_backend_arg(sys.argv[1:])
    sys.argv = [sys.argv[0], *forwarded_argv]
    from backends.torch.train_backend import main as backend_main

    backend_main()


if __name__ == "__main__":
    main()
