import numpy as np
from tqdm import tqdm

from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

# global config

BOARD_DIM = 3
DATA_DIR = "data/3x3"

# use class balancing on training set
USE_BALANCING = True   # you can flip this to False to match the test imbalance

# CSV base names
DATASETS = {
    "final": "hex3_final_8000_games",
    "mb2":   "hex3_mb2_8000_games",
    "mb5":   "hex3_mb5_8000_games",
}

# limit how many samples to use (set to None to use all)
USE_TRAIN_SAMPLES = 8000
USE_TEST_SAMPLES = 800

# epochs per dataset
EPOCHS = {
    "final": 15,
    "mb2":   15,
    "mb5":   15,
}

# separate TM configs per dataset
# (updated: lower clause counts + depth=2 to reduce overfitting and use graph structure)
TM_CONFIGS = {
    "final": {
        "number_of_clauses": 220,
        "T": 200,
        "s": 1.4,
        "depth": 1,
        "q": 1.0,
        "message_size": 64,
        "grid": (16 * 13, 1, 1),
        "block": (128, 1, 1),
    },
    "mb2": {
        "number_of_clauses": 500,
        "T": 490,
        "s": 2.0,
        "depth": 1,
        "q": 1.0,
        "message_size": 64,
        "grid": (16 * 13, 1, 1),
        "block": (128, 1, 1),
    },
    "mb5": {
        "number_of_clauses": 550,
        "T": 520,
        "s": 5.0,  # was 5; slightly gentler feedback
        "depth": 1,
        "q": 1.0,
        "message_size": 64,
        "grid": (16 * 13, 1, 1),
        "block": (128, 1, 1),
    },
}


# neighbors & graph building

def neighbors(idx: int):
    """Hex neighbors on a BOARD_DIM x BOARD_DIM board for a cell index 0..N-1."""
    r = idx // BOARD_DIM
    c = idx % BOARD_DIM
    nbrs = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, -1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < BOARD_DIM and 0 <= cc < BOARD_DIM:
            nbrs.append(rr * BOARD_DIM + cc)
    return nbrs


def node_type_name(r: int, c: int) -> str:
    on_top = (r == 0)
    on_bottom = (r == BOARD_DIM - 1)
    on_left = (c == 0)
    on_right = (c == BOARD_DIM - 1)

    if on_top and on_left:  return "TopLeft"
    if on_top and on_right: return "TopRight"
    if on_bottom and on_left:  return "BottomLeft"
    if on_bottom and on_right: return "BottomRight"
    if on_top:    return "Top"
    if on_bottom: return "Bottom"
    if on_left:   return "Left"
    if on_right:  return "Right"
    return "Middle"


def boards_to_graphs(boards: np.ndarray, template: Graphs = None) -> Graphs:
    """
    Convert a batch of boards (N, BOARD_DIM, BOARD_DIM) to a Graphs object.

    Each node gets:
      - content symbol: E, B, or W
      - position symbols: R0..R(BOARD_DIM-1), C0..C(BOARD_DIM-1)
      - node type: Top/Bottom/Left/Right/... and a special 'Center' for the middle cell

    If template is provided, reuse its encoding via init_with so that
    train and test share the same symbol/hypervector space.
    """
    N = boards.shape[0]

    # row / col symbols so the TM can learn location-sensitive patterns
    row_symbols = [f"R{r}" for r in range(BOARD_DIM)]
    col_symbols = [f"C{c}" for c in range(BOARD_DIM)]
    symbols = ["E", "B", "W"] + row_symbols + col_symbols  # 0: empty, 1: black, 2: white

    if template is None:
        graphs = Graphs(
            number_of_graphs=N,
            symbols=symbols,
            hypervector_size=64,
            hypervector_bits=2,
        )
    else:
        graphs = Graphs(
            number_of_graphs=N,
            init_with=template,
        )

    for g in range(N):
        graphs.set_number_of_graph_nodes(g, BOARD_DIM * BOARD_DIM)

    graphs.prepare_node_configuration()

    for g in tqdm(range(N), desc="Add nodes", unit="board"):
        for idx in range(BOARD_DIM * BOARD_DIM):
            r = idx // BOARD_DIM
            c = idx % BOARD_DIM
            val = boards[g, r, c]          # 0,1,2

            symbol = ["E", "B", "W"][val]
            deg = len(neighbors(idx))
            ntype = node_type_name(r, c)

            # Make the exact center have a unique type name
            if r == BOARD_DIM // 2 and c == BOARD_DIM // 2:
                ntype = "Center"

            graphs.add_graph_node(g, str(idx), deg, node_type_name=ntype)

            # Board content (empty / black / white)
            graphs.add_graph_node_property(g, str(idx), symbol)

            # Positional information (row & column)
            graphs.add_graph_node_property(g, str(idx), f"R{r}")
            graphs.add_graph_node_property(g, str(idx), f"C{c}")

    graphs.prepare_edge_configuration()
    for g in tqdm(range(N), desc="Add edges", unit="board"):
        for i in range(BOARD_DIM * BOARD_DIM):
            for nb in neighbors(i):
                graphs.add_graph_node_edge(g, str(i), str(nb), "adj")

    graphs.encode()
    return graphs


# csv loading

def load_csv_dataset(path: str):
    """Load dataset from CSV. Each row: cell_0,...,cell_(N-1),label."""
    raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    X_flat = raw[:, :-1]
    y = raw[:, -1].astype(np.uint32)

    boards = X_flat.reshape(-1, BOARD_DIM, BOARD_DIM).astype(np.int8)
    return boards, y


# training / eval for one dataset

def run_for_dataset(tag: str):
    dataset_base = DATASETS[tag]
    train_csv = f"{DATA_DIR}/{dataset_base}_train.csv"
    test_csv  = f"{DATA_DIR}/{dataset_base}_test.csv"
    n_epochs  = EPOCHS[tag]
    tm_params = TM_CONFIGS[tag]

    print(f"\n\n========== DATASET: {tag} ({dataset_base}) ==========")

    boards_train, y_train = load_csv_dataset(train_csv)
    boards_test,  y_test  = load_csv_dataset(test_csv)

    # optional subsampling of train/test
    if USE_TRAIN_SAMPLES is not None and USE_TRAIN_SAMPLES < len(boards_train):
        train_idx = np.random.permutation(len(boards_train))[:USE_TRAIN_SAMPLES]
        boards_train = boards_train[train_idx]
        y_train = y_train[train_idx]

    if USE_TEST_SAMPLES is not None and USE_TEST_SAMPLES < len(boards_test):
        test_idx = np.random.permutation(len(boards_test))[:USE_TEST_SAMPLES]
        boards_test = boards_test[test_idx]
        y_test = y_test[test_idx]

    print("Train:", boards_train.shape, "labels:", np.bincount(y_train))
    print("Test: ", boards_test.shape,  "labels:", np.bincount(y_test))

    if USE_BALANCING:
        idx0 = np.where(y_train == 0)[0]
        idx1 = np.where(y_train == 1)[0]
        n = min(len(idx0), len(idx1))

        if n == 0:
            raise RuntimeError(f"[{tag}] One class has zero samples in training set; cannot balance.")

        np.random.shuffle(idx0)
        np.random.shuffle(idx1)
        sel = np.concatenate([idx0[:n], idx1[:n]])
        np.random.shuffle(sel)

        boards_bal = boards_train[sel]
        y_bal = y_train[sel]

        print("Balanced train:", boards_bal.shape, np.bincount(y_bal))
    else:
        boards_bal = boards_train
        y_bal = y_train
        print("No balancing, using full train set:", boards_bal.shape, np.bincount(y_bal))

    # Encode graphs: train defines encoding, test reuses it
    print("\nEncoding TRAIN graphs...")
    graphs_train = boards_to_graphs(boards_bal)
    y_uint = y_bal.astype(np.uint32)

    print("\nEncoding TEST graphs...")
    graphs_test = boards_to_graphs(boards_test, template=graphs_train)

    tm = MultiClassGraphTsetlinMachine(**tm_params)
    try:
        backend = tm.get_backend()
        device = "GPU (CUDA)" if backend == 1 else "CPU"
    except Exception:
        device = "CPU"

    print(f"\nInitialized TM for [{tag}] — Running on: {device}")
    print("Training...")

    def train_one_epoch(tm, graphs, y_uint, epoch):
        if epoch == 1:
            tm.fit(graphs, y_uint, epochs=1, incremental=True)
            preds = tm.predict(graphs)
            acc = np.mean(preds == y_uint)
            print(f"Epoch {epoch:02d} accuracy: {acc:.4f}")
        else:
            w_before = tm.get_weights().copy()
            tm.fit(graphs, y_uint, epochs=1, incremental=True)
            w_after = tm.get_weights()
            diff = np.sum(w_before != w_after)

            preds = tm.predict(graphs)
            acc = np.mean(preds == y_uint)
            print(f"Epoch {epoch:02d} accuracy: {acc:.4f} | #weights changed: {diff}")

    for epoch in range(1, n_epochs + 1):
        train_one_epoch(tm, graphs_train, y_uint, epoch)

    print("\nEvaluating on TEST set...")
    preds_test = tm.predict(graphs_test)
    test_acc = np.mean(preds_test == y_test)
    print(f"[{tag}] TEST ACCURACY: {test_acc:.4f}")

    print("Some test predictions (pred, true):",
          list(zip(preds_test[:10], y_test[:10])))


# main

if __name__ == "__main__":
    for tag in ["final", "mb2", "mb5"]:
        run_for_dataset(tag)
