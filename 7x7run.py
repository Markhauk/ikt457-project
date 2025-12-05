import numpy as np
from tqdm import tqdm

from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

# global config

BOARD_DIM = 5
DATA_DIR = "data/7x7"

# CSV base names
DATASETS = {
    "final": "hex7_final_400000_games",
    "mb2":   "hex7_mb2_400000_games",
    "mb5":   "hex7_mb5_400000_games",
}

# limit how many samples to use (set to None to use all)
USE_TRAIN_SAMPLES = 200000
USE_TEST_SAMPLES = 20000

# epochs per dataset
EPOCHS = {
    "final": 10,
    "mb2":   10,
    "mb5":   10,
}

# separate TM configs per dataset
TM_CONFIGS = {
    "final": {
        "number_of_clauses": 6000,
        "T": 1500,
        "s": 1.0,
        "depth": 1,
        "q": 1.0,
        "message_size": 64,
        "grid": (16 * 13, 1, 1),
        "block": (128, 1, 1),
    },
    "mb2": {
        "number_of_clauses": 1200,
        "T": 150,
        "s": 9.0,
        "depth": 2,
        "q": 1.0,
        "message_size": 64,
        "grid": (16 * 13, 1, 1),
        "block": (128, 1, 1),
    },
    "mb5": {
        "number_of_clauses": 1200,
        "T": 150,
        "s": 10.0,
        "depth": 2,
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


def boards_to_graphs(boards: np.ndarray) -> Graphs:
    """Convert a batch of boards (N, BOARD_DIM, BOARD_DIM) to a Graphs object."""
    N = boards.shape[0]
    graphs = Graphs(
        number_of_graphs=N,
        symbols=["E", "B", "W"],  # 0: empty, 1: black, 2: white
        hypervector_size=64,
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

            graphs.add_graph_node(g, str(idx), deg, node_type_name=ntype)
            graphs.add_graph_node_property(g, str(idx), symbol)

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

    # balance training set (50/50)
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

    graphs_train = boards_to_graphs(boards_bal)
    y_uint = y_bal.astype(np.uint32)

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
#========================================================
#            print("\nInspecting clause outputs for 3 samples...")
#            X_clause, class_sums = tm.transform(graphs)
#            print("  transformed_X shape:", X_clause.shape)
#            print("  class_sums shape:", class_sums.shape)
#
#            for i in range(3):
#                print(
#                    f"Sample {i}, true_label={y_uint[i]}, "
#                    f"class_sum={class_sums[i]}"
#                )
#                print("  clause outputs (first 40):", X_clause[i, :40])
#===========================================================================        
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
    graphs_test = boards_to_graphs(boards_test)
    preds_test = tm.predict(graphs_test)
    test_acc = np.mean(preds_test == y_test)
    print(f"[{tag}] TEST ACCURACY: {test_acc:.4f}")

    print("Some test predictions (pred, true):",
          list(zip(preds_test[:10], y_test[:10])))


# main

if __name__ == "__main__":
    for tag in ["final", "mb2", "mb5"]:
        run_for_dataset(tag)
