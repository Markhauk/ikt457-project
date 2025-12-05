import os
import csv
import random
import subprocess
import numpy as np
from pathlib import Path

BOARD_DIM = 7          # set this before calling load_from_c
EXE_PATH = "hex/hex"   # path to your compiled C simulator


def load_from_c(exe_path: str, num_games: int, moves_back: int):
    """
    Generic loader: returns boards, y for a specific MOVES_BACK value.
    moves_back: 0 (final), 2, or 5.
    """
    N_CELLS = BOARD_DIM * BOARD_DIM

    cmd = [exe_path, str(num_games)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    boards = []
    winners = []

    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("MOVES_BACK"):
            continue

        parts = line.split()
        try:
            mb = int(parts[1])
        except Exception:
            continue

        if mb != moves_back:
            continue

        if "WINNER" not in parts:
            continue
        win_idx = parts.index("WINNER")

        cell_strs = parts[3:3 + N_CELLS]
        if len(cell_strs) != N_CELLS:
            continue

        try:
            cell_vals = list(map(int, cell_strs))
            winner = int(parts[win_idx + 1])
        except Exception:
            continue

        board = np.array(cell_vals, dtype=np.int8).reshape(BOARD_DIM, BOARD_DIM)
        boards.append(board)
        winners.append(winner)

    proc.wait()

    if len(boards) == 0:
        raise RuntimeError(f"No MOVES_BACK {moves_back} DATA lines parsed from C output!")

    boards = np.stack(boards, axis=0)
    y = np.array(winners, dtype=np.int32)
    return boards, y


def stratified_train_test_split(boards, y, test_ratio=0.1, rng_seed=42):
    """
    Simple stratified split on our (boards, y).

    Returns:
        boards_train, y_train, boards_test, y_test
    """
    assert boards.shape[0] == y.shape[0]
    n_total = boards.shape[0]

    rng = np.random.default_rng(rng_seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    rng.shuffle(idx0)
    rng.shuffle(idx1)

    n0 = len(idx0)
    n1 = len(idx1)

    n_test0 = int(round(test_ratio * n0))
    n_test1 = int(round(test_ratio * n1))

    test_idx = np.concatenate([idx0[:n_test0], idx1[:n_test1]])
    train_idx = np.concatenate([idx0[n_test0:], idx1[n_test1:]])

    rng.shuffle(test_idx)
    rng.shuffle(train_idx)

    boards_train = boards[train_idx]
    y_train = y[train_idx]
    boards_test = boards[test_idx]
    y_test = y[test_idx]

    return boards_train, y_train, boards_test, y_test


def save_dataset_csv(path, boards, y):
    """
    Save boards + labels to CSV.
    Each row: cell_0, cell_1, ..., cell_(N-1), label
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    N = boards.shape[0]
    N_cells = boards.shape[1] * boards.shape[2]

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        # header (optional)
        header = [f"cell_{i}" for i in range(N_cells)] + ["label"]
        writer.writerow(header)

        for i in range(N):
            flat = boards[i].reshape(-1)
            row = list(map(int, flat)) + [int(y[i])]
            writer.writerow(row)


def generate_hex_dataset(board_dim: int,
                         num_games: int,
                         exe_path: str,
                         moves_back: int,
                         base_name: str,
                         test_ratio: float = 0.1):
    """
    High-level helper:
      - sets BOARD_DIM
      - calls C simulator
      - does stratified split
      - writes X_train/X_test CSVs.

    base_name: e.g. "hex5_final", "hex5_m2", "hex7_m5"
    """
    global BOARD_DIM
    BOARD_DIM = board_dim

    print(f"Generating dataset for BOARD_DIM={BOARD_DIM}, "
          f"moves_back={moves_back}, num_games={num_games}")
    boards, y = load_from_c(exe_path, num_games, moves_back)

    print("  Total samples:", boards.shape[0])
    print("  Class counts:", np.bincount(y))

    boards_train, y_train, boards_test, y_test = stratified_train_test_split(
        boards, y, test_ratio=test_ratio, rng_seed=42
    )

    print("  Train shape:", boards_train.shape, np.bincount(y_train))
    print("  Test  shape:", boards_test.shape, np.bincount(y_test))

    # Save CSVs
    train_csv = f"{base_name}_train.csv"
    test_csv = f"{base_name}_test.csv"

    save_dataset_csv(train_csv, boards_train, y_train)
    save_dataset_csv(test_csv, boards_test, y_test)

    # Optional tiny log
    with open(f"{base_name}_log.txt", "w") as log:
        log.write(f"BOARD_DIM: {BOARD_DIM}\n")
        log.write(f"NUM_GAMES: {num_games}\n")
        log.write(f"MOVES_BACK: {moves_back}\n")
        log.write(f"test_ratio: {test_ratio}\n")
        log.write(f"Total: {boards.shape[0]}, class_counts: {np.bincount(y)}\n")
        log.write(f"Train: {boards_train.shape[0]}, class_counts: {np.bincount(y_train)}\n")
        log.write(f"Test:  {boards_test.shape[0]}, class_counts: {np.bincount(y_test)}\n")

    print(f"  Saved: {train_csv}, {test_csv}")


if __name__ == "__main__":
    generate_hex_dataset(
        board_dim=3,
        num_games=8000,
        exe_path=EXE_PATH,
        moves_back=0,            # final
        base_name="hex3_final_8000_games",
        test_ratio=0.1
    )

    generate_hex_dataset(
        board_dim=3,
        num_games=8000,
        exe_path=EXE_PATH,
        moves_back=2,            #mb2
        base_name="hex3_mb2_8000_games",
        test_ratio=0.1
    )

    generate_hex_dataset(
        board_dim=3,
        num_games=8000,
        exe_path=EXE_PATH,
        moves_back=5,            #mb5
        base_name="hex3_mb5_8000_games",
        test_ratio=0.1
    )