"""Top-level GUI entrypoint with backend dispatch."""

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
                raise SystemExit("--backend requires a value: numpy or torch")
            backend = argv[index + 1].strip().lower()
            skip_next = True
            continue
        if arg.startswith("--backend="):
            backend = arg.split("=", 1)[1].strip().lower()
            continue
        out.append(arg)
    if backend not in {"numpy", "torch"}:
        raise SystemExit(f"Unsupported backend: {backend}")
    return backend, out


def main() -> None:
    backend, forwarded_argv = _pop_backend_arg(sys.argv[1:])
    if any(arg in {"-h", "--help"} for arg in forwarded_argv):
        print("Global option: --backend {torch,numpy} (default: torch)")
    sys.argv = [sys.argv[0], *forwarded_argv]
    try:
        if backend == "numpy":
            from backends.numpy.gui_backend import main as backend_main
        else:
            from backends.torch.gui_backend import main as backend_main
    except Exception as exc:
        if backend == "torch":
            raise SystemExit(f"Failed to load the PyTorch backend: {exc}. Use `py -3.13 gui.py` or `--backend numpy`.") from exc
        raise
    backend_main()


if __name__ == "__main__":
    main()
