from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def savefig(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
