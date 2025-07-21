import os, sys
from pathlib import Path
import subprocess
import tomllib
from typing import Union, Optional
from types import SimpleNamespace


def git_root() -> Path:
    """
    Returns the root directory of the git repo this file is in.
    """

    path = Path(
        subprocess.Popen(
            ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE
        )
        .communicate()[0]
        .rstrip()
        .decode("utf-8")
    )

    if path.exists:
        return path

    for p in Path(__file__).parents:
        if (p / ".git").exists():
            return p

    raise RuntimeError("Could not determine git repo root. Are we in a git repo?")


def load_config_file(path: Path | str) -> SimpleNamespace:
    """
    Load TOML config file with every dictionary converted to a SimpleNamespace.
    """

    def parse(d):
        x = SimpleNamespace()
        for k, v in d.items():
            setattr(x, k, parse(v)) if isinstance(v, dict) else setattr(x, k, v)
        return x

    with open(path, "rb") as fp:
        cfg = tomllib.load(fp)

    return parse(cfg)
