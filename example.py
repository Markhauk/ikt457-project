import subprocess
import numpy as np

from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from collections import Counter
from tqdm import tqdm


# ---------- BOARD → GRAPH ENCODING ----------

BOARD_DIM = 3  # must match hex.c

def neighbors(idx: int):
    """Hex neighbors on a BOARD_DIM x BOARD_DIM board for a cell index 0..N-1."""
    r = idx // BOARD_DIM
    c = idx % BOARD_DIM
    nbrs = []
    # Standard hex neighbors: up, down, left, right, up-right, down-left
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
    N = boards.shape[0]
    graphs = Graphs(
        number_of_graphs=N,
        symbols=["E", "B", "W"],  # 0: empty, 1: black, 2: white
        hypervector_size=64
    )

    print("Setting graph node counts...")
    for g in range(N):
        graphs.set_number_of_graph_nodes(g, BOARD_DIM * BOARD_DIM)

    graphs.prepare_node_configuration()

    print("Adding nodes and properties...")
    for g in tqdm(
        range(N),
        desc="Nodes/properties",
        unit="board",
        ncols=80,
        leave=False
    ):
        for idx in range(BOARD_DIM * BOARD_DIM):
            r = idx // BOARD_DIM
            c = idx % BOARD_DIM
            val = boards[g, r, c]

            symbol = ["E", "B", "W"][val]
            deg = len(neighbors(idx))

            # optional node type, if you still use it:
            # ntype = node_type_name(r, c)
            # graphs.add_graph_node(g, str(idx), deg, node_type_name=ntype)
            graphs.add_graph_node(g, str(idx), deg)
            graphs.add_graph_node_property(g, str(idx), symbol)

    print("Preparing edges...")
    graphs.prepare_edge_configuration()

    print("Adding edges...")
    for g in tqdm(
        range(N),
        desc="Edges",
        unit="board",
        ncols=80,
        leave=False
    ):
        for i in range(BOARD_DIM * BOARD_DIM):
            for nb in neighbors(i):
                graphs.add_graph_node_edge(g, str(i), str(nb), "adj")

    print("Encoding...")
    graphs.encode()
    return graphs





def train_one_epoch_with_weight_change(tm, graphs, y):
    """
    Train the GTM for one epoch and print how many clause weights changed.

    On the very first call, number_of_outputs is not defined yet, so we just
    run one epoch to initialize everything.
    """
    if not hasattr(tm, "number_of_outputs"):
        # First time: do one epoch to initialize internal structures.
        tm.fit(graphs, y, epochs=1, incremental=True)
        print("  (first epoch: model initialized, weight diff not measured)")
        return

    # After first epoch we can inspect weights
    w_before = tm.get_weights().copy()
    tm.fit(graphs, y, epochs=1, incremental=True)
    w_after = tm.get_weights()
    diff = np.sum(w_before != w_after)
    print(f"  #weights changed this epoch: {diff}")


# ---------- C INTERFACE ----------

def load_from_c(exe_path: str, num_games: int):
    N_CELLS = BOARD_DIM * BOARD_DIM

    cmd = [exe_path, str(num_games)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    boards_0, boards_2, boards_5 = [], [], []
    winners_0, winners_2, winners_5 = [], [], []

    for line in proc.stdout:
        line = line.strip()
        if not line.startswith("MOVES_BACK"):
            continue

        parts = line.split()
        # MOVES_BACK k DATA v0 ... v(N-1) WINNER w
        try:
            moves_back = int(parts[1])
        except Exception:
            continue

        if "WINNER" not in parts:
            continue
        win_idx = parts.index("WINNER")

        # cells start after 'MOVES_BACK', <k>, 'DATA'
        cell_strs = parts[3:win_idx]
        if len(cell_strs) != N_CELLS:
            continue

        try:
            cell_vals = list(map(int, cell_strs))
            winner = int(parts[win_idx + 1])
        except Exception:
            continue

        board = np.array(cell_vals, dtype=np.int8).reshape(BOARD_DIM, BOARD_DIM)

        if moves_back == 0:
            boards_0.append(board); winners_0.append(winner)
        elif moves_back == 2:
            boards_2.append(board); winners_2.append(winner)
        elif moves_back == 5:
            boards_5.append(board); winners_5.append(winner)

    proc.wait()

    def stack_or_empty(lst_b, lst_y):
        if len(lst_b) == 0:
            return np.empty((0, BOARD_DIM, BOARD_DIM), dtype=np.int8), np.empty((0,), dtype=np.int32)
        return np.stack(lst_b, axis=0), np.array(lst_y, dtype=np.int32)

    return (
        stack_or_empty(boards_0, winners_0),
        stack_or_empty(boards_2, winners_2),
        stack_or_empty(boards_5, winners_5),
    )


# ---------- SIMPLE FILTER ----------

def filter_min_moves(boards, y, min_non_zero=1):
    """Remove trivial games with too few non-empty cells."""
    if boards.shape[0] == 0:
        return boards, y
    non_zero = np.sum(boards != 0, axis=(1, 2))
    mask = non_zero >= min_non_zero
    return boards[mask], y[mask]

if __name__ == "__main__":
    exe_path = "hex/hex"
    num_games = 10000   # your current value

    print("Running C simulator and loading data...")
    (boards_final, y_final), (boards_m2, y_m2), (boards_m5, y_m5) = load_from_c(exe_path, num_games)

    print("Shapes:")
    print("  final:             ", boards_final.shape, y_final.shape)
    print("  -2 moves:          ", boards_m2.shape, y_m2.shape)
    print("  -5 moves:          ", boards_m5.shape, y_m5.shape)

    # --- stats on final positions ---
    print("Class distribution (winners) at end of game:")
    print("  mean label:", np.mean(y_final))
    print("  count 0:", np.sum(y_final == 0))
    print("  count 1:", np.sum(y_final == 1))

    baseline = max(np.mean(y_final == 0), np.mean(y_final == 1))
    print("Majority-class baseline accuracy (final boards):", baseline)

    # --- stats on 5-moves-before boards ---
    print("\nClass distribution 5 moves before end:")
    if len(y_m5) > 0:
        print("  mean label:", np.mean(y_m5))
        print("  count 0:", np.sum(y_m5 == 0))
        print("  count 1:", np.sum(y_m5 == 1))
    else:
        print("  (no 5-move boards returned!)")

    # =====================================================
    # CHOOSE WHAT TO TRAIN ON
    # =====================================================
    TRAIN_SOURCE = "final"   # or "m5"

    if TRAIN_SOURCE == "final":
        all_boards = boards_final
        all_y = y_final
        print("\nTraining source: FINAL boards")
    elif TRAIN_SOURCE == "m5":
        all_boards = boards_m5
        all_y = y_m5
        print("\nTraining source: 5-moves-before-end boards")
    else:
        raise ValueError("Unsupported TRAIN_SOURCE")

    # Filter trivially empty boards
    all_boards, all_y = filter_min_moves(all_boards, all_y, min_non_zero=1)

    if all_boards.shape[0] == 0:
        raise RuntimeError("No boards after filtering – cannot proceed.")

    # =====================================================
    # TRAIN / TEST SPLIT (on the chosen source only)
    # =====================================================
    rng = np.random.default_rng(seed=42)
    indices = np.arange(all_boards.shape[0])
    rng.shuffle(indices)

    split = int(0.8 * len(indices))  # 80% train, 20% test
    train_idx = indices[:split]
    test_idx  = indices[split:]

    train_boards = all_boards[train_idx]
    train_y = all_y[train_idx]

    test_boards = all_boards[test_idx]
    test_y = all_y[test_idx]

    # =====================================================
    # BALANCING (still OFF for now)
    # =====================================================
    BALANCE = False

    if BALANCE:
        print("\n[Balancing] Before balancing:")
        print("  class 0 count:", np.sum(train_y == 0))
        print("  class 1 count:", np.sum(train_y == 1))

        idx0 = np.where(train_y == 0)[0]
        idx1 = np.where(train_y == 1)[0]

        if len(idx0) == 0 or len(idx1) == 0:
            print("  [Balancing] WARNING: only one class present, skip balancing.")
            train_boards_bal = train_boards
            train_y_bal = train_y
        else:
            n = min(len(idx0), len(idx1))
            rng.shuffle(idx0)
            rng.shuffle(idx1)
            sel = np.concatenate([idx0[:n], idx1[:n]])
            rng.shuffle(sel)

            train_boards_bal = train_boards[sel]
            train_y_bal = train_y[sel]

            print("[Balancing] After balancing:")
            print("  class 0 count:", np.sum(train_y_bal == 0))
            print("  class 1 count:", np.sum(train_y_bal == 1))
    else:
        train_boards_bal = train_boards
        train_y_bal = train_y

    print("\nTraining data shape:", train_boards_bal.shape, train_y_bal.shape)
    print("Training class distribution:")
    print("  count 0:", np.sum(train_y_bal == 0))
    print("  count 1:", np.sum(train_y_bal == 1))

    if train_boards_bal.shape[0] == 0:
        raise RuntimeError("Still have 0 training boards after all checks – cannot train.")

    # ==================================
    # LABEL FLIP EXPERIMENT (TURNED OFF NOW)
    # ==================================
    FLIP_LABELS_FOR_GTM = False  # <- important: off

    y_train_orig = train_y_bal.astype(np.uint32)
    y_test_orig  = test_y.astype(np.uint32)

    if FLIP_LABELS_FOR_GTM:
        print("\n[Label Flip] Training GTM on FLIPPED labels (0<->1)")
        y_train = 1 - y_train_orig
        y_test  = 1 - y_test_orig
    else:
        print("\n[Label Flip] Training GTM on ORIGINAL labels")
        y_train = y_train_orig
        y_test  = y_test_orig

    # ==================================
    # 2) Convert TRAIN and TEST boards to graphs
    #    *** KEY FIX: use same random seed so encoding is consistent ***
    # ==================================
    SYMBOL_SEED = 12345

    print("Converting TRAINING boards to graphs...")
    np.random.seed(SYMBOL_SEED)
    graphs_train = boards_to_graphs(train_boards_bal)

    print("Converting TEST boards to graphs...")
    np.random.seed(SYMBOL_SEED)
    graphs_test = boards_to_graphs(test_boards)

    # Quick sanity check on graph encoding
    print("Graph debug (train):")
    print("  number_of_graphs:", graphs_train.number_of_graphs)
    print("  max_number_of_graph_nodes:", graphs_train.max_number_of_graph_nodes)
    print("  number_of_node_types:", graphs_train.number_of_node_types())
    print("  edge_type_id:", graphs_train.edge_type_id)

    # ==================================
    # 3) Initialize GTM
    # ==================================
    print("Initializing Graph Tsetlin Machine...")
    tm = MultiClassGraphTsetlinMachine(
        number_of_clauses=400,
        T=10,
        s=5.0,
        depth=1,
        q=1.0,
        message_size=64,
        grid=(32, 1, 1),
        block=(64, 1, 1),
    )

    # ==================================
    # 4) Training loop with monitoring + early stopping
    # ==================================
    print("Training TM...")
    epochs = 5
    best_test_acc = -1.0
    best_state = None

    from collections import Counter

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):
        tm.fit(graphs_train, y_train, epochs=1, incremental=True)
    
        preds_train_model = tm.predict(graphs_train)
        acc_train_model = np.mean(preds_train_model == y_train)
    
        # convert to original-label space if needed
        preds_train_orig = 1 - preds_train_model if FLIP_LABELS_FOR_GTM else preds_train_model
        acc_train_orig = np.mean(preds_train_orig == y_train_orig)
    
        # Debug output ONLY on epoch 1
        if epoch == 1:
            print("\nInspecting clause outputs for 3 samples...")
            X_clause, class_sums = tm.transform(graphs_train)
            print("  transformed_X shape:", X_clause.shape)
            print("  class_sums shape:", class_sums.shape)
    
            for i in range(3):
                print(f"Sample {i}, true_label={y_train_orig[i]}, train_label_used={y_train[i]}, class_sum={class_sums[i]}")
                print("  clause outputs (first 40):", X_clause[i, :40])
    
        # ---- test metrics ----
        preds_test_model = tm.predict(graphs_test)
        acc_test_model = np.mean(preds_test_model == y_test)
        preds_test_orig = 1 - preds_test_model if FLIP_LABELS_FOR_GTM else preds_test_model
        acc_test_orig = np.mean(preds_test_orig == y_test_orig)
    
        # Display compact epoch status
        tqdm.write(
            f"Epoch {epoch:02}/{epochs}  "
            f"Train={acc_train_orig:.3f}  "
            f"Test={acc_test_orig:.3f}  "
            f"Dist=[0:{np.mean(preds_test_orig==0):.2f} 1:{np.mean(preds_test_orig==1):.2f}]"
        )
    
        # Update best and show confusion matrix ONLY when improving
        if acc_test_orig > best_test_acc:
            best_test_acc = acc_test_orig
            best_state = tm.get_state()
            tqdm.write(f"  ** New best test accuracy: {best_test_acc:.4f} **")
    
    if best_state is not None:
        tm.set_state(best_state)
        print(f"\nRestored best model with test accuracy (original labels): {best_test_acc:.4f}")



    # ==================================
    # 5) Final test evaluation (ORIGINAL labels)
    # ==================================
    print("\nFinal evaluation on held-out TEST set of", TRAIN_SOURCE, "boards...")

    preds_test_model = tm.predict(graphs_test)
    preds_test_orig = 1 - preds_test_model if FLIP_LABELS_FOR_GTM else preds_test_model
    acc_test_orig = np.mean(preds_test_orig == y_test_orig)

    print(f"Test accuracy on {TRAIN_SOURCE} boards (original labels): {acc_test_orig:.3f}")
    print("  Preds class distribution (test, ORIGINAL labels):",
          np.mean(preds_test_orig == 0),
          np.mean(preds_test_orig == 1))
    print("  True class distribution (test):",
          np.mean(y_test_orig == 0),
          np.mean(y_test_orig == 1))

    # ==================================
    # 6) Extra diagnostics on ALL boards (not split)
    # ==================================
    print("\nEvaluating on ALL boards (no train/test split here, just diagnostics)...")

    np.random.seed(SYMBOL_SEED)
    graphs_final_all = boards_to_graphs(boards_final)
    np.random.seed(SYMBOL_SEED)
    graphs_m2_all    = boards_to_graphs(boards_m2)
    np.random.seed(SYMBOL_SEED)
    graphs_m5_all    = boards_to_graphs(boards_m5)

    preds_final_all_model = tm.predict(graphs_final_all)
    preds_m2_all_model    = tm.predict(graphs_m2_all)
    preds_m5_all_model    = tm.predict(graphs_m5_all)

    preds_final_all = 1 - preds_final_all_model if FLIP_LABELS_FOR_GTM else preds_final_all_model
    preds_m2_all    = 1 - preds_m2_all_model    if FLIP_LABELS_FOR_GTM else preds_m2_all_model
    preds_m5_all    = 1 - preds_m5_all_model    if FLIP_LABELS_FOR_GTM else preds_m5_all_model

    acc_0 = np.mean(preds_final_all == y_final)
    acc_2 = np.mean(preds_m2_all == y_m2)
    acc_5 = np.mean(preds_m5_all == y_m5)

    print(f"Accuracy at end of game (ALL, original labels):         {acc_0:.3f}")
    print("  Preds class distribution (final ALL, ORIGINAL labels):",
          np.mean(preds_final_all == 0),
          np.mean(preds_final_all == 1))

    print(f"Accuracy two moves before end (ALL, original labels):   {acc_2:.3f}")
    print("  Preds class distribution (m2 ALL, ORIGINAL labels):",
          np.mean(preds_m2_all == 0),
          np.mean(preds_m2_all == 1))

    print(f"Accuracy five moves before end (ALL, original labels):  {acc_5:.3f}")
    print("  Preds class distribution (m5 ALL, ORIGINAL labels):",
          np.mean(preds_m5_all == 0),
          np.mean(preds_m5_all == 1))

    print("\nSome predictions vs labels (final boards, ALL, ORIGINAL labels):")
    for i in range(10):
        print(i, "pred:", preds_final_all[i], "label:", y_final[i])

