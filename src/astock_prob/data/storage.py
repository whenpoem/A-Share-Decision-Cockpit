from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.errors import EmptyDataError


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    return target


def load_frame(path: str | Path, parse_dates: Optional[list[str]] = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except EmptyDataError:
        return pd.DataFrame()
