#!/usr/bin/env python3
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

CSV_DIRECTORY = Path(__file__).parent / "raw_data/csv"
NPZ_DIRECTORY = Path(__file__).parent / "raw_data"

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def csv_to_npz(csv_file: str | Path) -> str:
    """
    Read 'B' and 'I' from a CSV and save as a compressed .npz.
    Raises 
        * FileNotFoundError: if not in CSV_DIRECTORY
        * ValueError:        for missing B/I columns.
    """

    csv_path = CSV_DIRECTORY / csv_file
    if not csv_path.is_file(): 
        logger.error("CSV not found in %s: %s", CSV_DIRECTORY, csv_path)
        raise FileNotFoundError
    try:
        df = pd.read_csv(csv_path, usecols=["B", "I"]) # type: ignore
    except ValueError as e:
        logger.error("CSV missing required columns B,I: %s", csv_path)
        raise ValueError("CSV must contain 'B' and 'I' columns") from e

    B, I = df["B"].to_numpy(), df["I"].to_numpy()
    NPZ_DIRECTORY.mkdir(parents=True, exist_ok=True)
    npz_path = NPZ_DIRECTORY / f"{csv_path.stem}.npz"

    np.savez_compressed(npz_path, B=B, I=I)
    logger.info("Saved NPZ to %s", npz_path)
    return npz_path.name

def npz_to_arrays(npz_file: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load 'B' and 'I' from a .npz file.
    Raises 
        * FileNotFoundError: if not in NPZ_DIRECTORY
        * KeyError:          if arrays missing.
    """

    npz_path = NPZ_DIRECTORY / npz_file 
    if not npz_path.is_file():
        logger.error("NPZ not found in %s: %s", NPZ_DIRECTORY, npz_path)
        raise FileNotFoundError
    with np.load(npz_path) as data:
        try:
            B = data["B"]
            I = data["I"]
        except KeyError as e:
            logger.exception("NPZ missing B/I arrays: %s", npz_path)
            raise e

    return B, I

if __name__ == "__main__":
    csv_file = "2.5V-1.5G.csv"
    npz_file = csv_to_npz(csv_file)
    B, I = npz_to_arrays(npz_file)
    logger.info("Loaded arrays: B (%d), I (%d)", B.size, I.size)

