import subprocess
import numpy as np

from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from tqdm import tqdm


BOARD_DIM = 5               
EXE_PATH = "hex/hex"         
NUM_GAMES = 100000          
N_EPOCHS = 5                

#  NEIGHBORS & GRAPH BUILDING

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


def boards_to_graphs(boards: np.ndarray) -> Graphs:
    """
    Convert a batch of Hex boards of shape (N, BOARD_DIM, BOARD_DIM)
    into a Graphs object, with node types (position) + properties (E/B/W)
    and adjacency edges.
    """
    N = boards.shape[0]
    graphs = Graphs(
        number_of_graphs=N,
        symbols=["E", "B", "W"],  # 0: empty, 1: black, 2: white
        hypervector_size=64,
    )

    # 1) Set number of nodes per graph
    for g in range(N):
        graphs.set_number_of_graph_nodes(g, BOARD_DIM * BOARD_DIM)

    # 2) Prepare node configuration
    graphs.prepare_node_configuration()

    # 3) Add nodes and node properties (with node_type_name)
    for g in tqdm(range(N), desc="Add nodes", unit="board"):
        for idx in range(BOARD_DIM * BOARD_DIM):
            r = idx // BOARD_DIM
            c = idx % BOARD_DIM
            val = boards[g, r, c]          # 0,1,2

            symbol = ["E", "B", "W"][val]   # map to symbol
            deg = len(neighbors(idx))
            ntype = node_type_name(r, c)   # <-- POSITION TYPE

            graphs.add_graph_node(g, str(idx), deg, node_type_name=ntype)
            graphs.add_graph_node_property(g, str(idx), symbol)

    # 4) Prepare edge configuration and add edges
    graphs.prepare_edge_configuration()
    for g in tqdm(range(N), desc="Add edges", unit="board"):
        for i in range(BOARD_DIM * BOARD_DIM):
            for nb in neighbors(i):
                graphs.add_graph_node_edge(g, str(i), str(nb), "adj")

    # 5) Encode
    graphs.encode()
    return graphs


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


#  C INTERFACE

def load_final_from_c(exe_path: str, num_games: int):
    """
    Only use MOVES_BACK 0 lines (final positions).
    Format:
        MOVES_BACK 0 DATA v0 ... v8 WINNER w
    Returns:
        boards: (N, 3, 3)
        y:      (N,)
    """
    N_CELLS = BOARD_DIM * BOARD_DIM

    cmd = [exe_path, str(num_games)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    boards = []
    winners = []

    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("MOVES_BACK"):
            continue

        parts = line.split()
        # we want final boards (moves_back == 0)
        try:
            moves_back = int(parts[1])
        except Exception:
            continue
        if moves_back != 0:
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
        raise RuntimeError("No MOVES_BACK 0 DATA lines parsed from C output!")

    boards = np.stack(boards, axis=0)
    y = np.array(winners, dtype=np.uint32)
    return boards, y


#  MAIN (DEBUG / WEIGHTED TRAINING)


if __name__ == "__main__":
    print("Running C simulator and loading x x final positions...")
    boards, y = load_final_from_c(EXE_PATH, NUM_GAMES)
    print("Loaded:", boards.shape, "labels:", np.bincount(y))

    # --- balance data ---
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n = min(len(idx0), len(idx1))

    if n == 0:
        raise RuntimeError("One class has zero samples, cannot balance.")

    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    sel = np.concatenate([idx0[:n], idx1[:n]])
    np.random.shuffle(sel)

    boards_bal = boards[sel]
    y_bal = y[sel]

    print("Balanced:", boards_bal.shape, np.bincount(y_bal))

    #  CHOOSE WHAT LABELS TO TRAIN ON
    #    "winner"        -> original C-simulator winners
    #    "center_black"  -> 1 if center cell == 1 (black), else 0
    LABEL_MODE = "center_black"   # <-- try "winner" later

    if LABEL_MODE == "center_black":
        # center of 3x3 is (1,1)
        center_vals = boards_bal[:, 1, 1]        # shape (N,)
        y_bal = (center_vals == 1).astype(np.uint32)
        print("\nUsing TOY LABEL TASK: center_black")
        print("  New label distribution:", np.bincount(y_bal))

    elif LABEL_MODE == "winner":
        print("\nUsing REAL LABEL TASK: game winner from C")

    else:
        raise ValueError("Unknown LABEL_MODE")

    
    if LABEL_MODE == "center_black":
        # center of 3x3 is (1,1)
        center_vals = boards_bal[:, 1, 1]        # shape (N,)
        y_bal = (center_vals == 1).astype(np.uint32)
        print("Using TOY LABEL TASK: center_black")
        print("  New label distribution:", np.bincount(y_bal))
    elif LABEL_MODE == "winner":
        print("Using REAL LABEL TASK: game winner from C")
    else:
        raise ValueError("Unknown LABEL_MODE")


    # --- build graphs ---
    graphs = boards_to_graphs(boards_bal)

    # --- init GTM (with depth=2 for edges) ---
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=600,   
        T=5,                    
        s=5.0,                   
        depth=2,                 
        q=1.0,
        message_size=64,
        grid=(16*13, 1, 1),
        block=(128, 1, 1),
    )
    try:
        backend = tm.get_backend()
        if backend == 1:
            device = "GPU (CUDA)"
        else:
            device = "CPU"
    except:
        device = "CPU"
    
    print(f"\nInitialized TM — Running on: {device}")

    print("\nInitialization of sparse structure.")
    print("Training...")

    # --- helper: measure weight change ---
    def train_one_epoch(tm, graphs, y_uint, epoch):
        if epoch == 1:
            tm.fit(graphs, y_uint, epochs=1, incremental=True)
            preds = tm.predict(graphs)
            acc = np.mean(preds == y_uint)
            print(f"Epoch {epoch:02d} accuracy: {acc:.4f}")

            # inspect clause outputs & class sums once
            print("\nInspecting clause outputs for 3 samples...")
            X_clause, class_sums = tm.transform(graphs)
            print("  transformed_X shape:", X_clause.shape)
            print("  class_sums shape:", class_sums.shape)

            for i in range(3):
                print(
                    f"Sample {i}, true_label={y_uint[i]}, "
                    f"class_sum={class_sums[i]}"
                )
                print("  clause outputs (first 40):", X_clause[i, :40])

        else:
            w_before = tm.get_weights().copy()
            tm.fit(graphs, y_uint, epochs=1, incremental=True)
            w_after = tm.get_weights()
            diff = np.sum(w_before != w_after)

            preds = tm.predict(graphs)
            acc = np.mean(preds == y_uint)
            print(f"Epoch {epoch:02d} accuracy: {acc:.4f} | #weights changed: {diff}")

    # --- training loop ---
    y_uint = y_bal.astype(np.uint32)

    for epoch in range(1, N_EPOCHS + 1):
        train_one_epoch(tm, graphs, y_uint, epoch)

    # --- final evaluation (on the same balanced set) ---
    preds = tm.predict(graphs)
    final_acc = np.mean(preds == y_uint)
    print("\nFINAL ACC (balanced set):", final_acc)

    print("Pred sample (pred, true):", list(zip(preds[:10], y_uint[:10])))
