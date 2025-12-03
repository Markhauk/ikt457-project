import numpy as np
from tmu.models.classification.vanilla_classifier import TMClassifier

# 1. Build your dataset (X = boards, y = winner)
X_list = []
y_list = []  # 0 = Black wins, 1 = White wins

# Example: you should replace this with real data from Kaggle or your generator
example_boards = [
    ([
        "B.....",
        ".B....",
        "..B...",
        "......",
        "......",
        "......",
     ], 0),   # Black wins
    ([
        "W.....",
        ".W....",
        "..W...",
        "......",
        "......",
        "......",
     ], 1),   # White wins
]

for board, winner in example_boards:
    X_list.append(encode_board(board))
    y_list.append(winner)

X = np.vstack(X_list)   # shape: (num_examples, num_features)
y = np.array(y_list)    # shape: (num_examples,)

# 2. Create a Tsetlin Machine
num_clauses = 100       # start small so it’s fast and easy to understand
T = 80                  # voting margin (often ~80% of num_clauses)
s = 10.0                # specificity parameter

tm = TMClassifier(
    number_of_clauses=num_clauses,
    T=T,
    s=s,
    platform='CPU',     # or 'GPU' if available
)

# 3. Train
epochs = 50
tm.fit(X, y, epochs=epochs)

# 4. Predict on new boards
test_board = [
    "B.....",
    ".B....",
    "..B...",
    "......",
    "......",
    "......",
]

x_test = encode_board(test_board).reshape(1, -1)
pred = tm.predict(x_test)[0]   # 0 or 1

print("Predicted winner:", "Black" if pred == 0 else "White")
