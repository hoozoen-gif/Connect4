import tkinter as tk
import numpy as np
import torch

from Connect4Game import Connect4
from model import DQN

# =====================================
# SETTINGS
# =====================================

ROWS = 6
COLS = 7
CELL_SIZE = 100

PLAYER = 1
AI = -1

MODEL_PATH = "models/current_best_model.pth"

# =====================================
# LOAD MODEL
# =====================================

device = torch.device("cpu")

model = DQN(
    nrow=ROWS,
    ncol=COLS,
    n_actions=COLS
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["online_model"]
)
model.eval()


# =====================================
# GAME
# =====================================

game = Connect4(ROWS, COLS)


# =====================================
# TKINTER WINDOW
# =====================================

window = tk.Tk()
window.title("Connect4 RL AI")

canvas = tk.Canvas(
    window,
    width=COLS * CELL_SIZE,
    height=ROWS * CELL_SIZE,
    bg="blue"
)

canvas.pack()

status_label = tk.Label(
    window,
    text="Your Turn",
    font=("Arial", 16)
)

status_label.pack()


# =====================================
# DRAW BOARD
# =====================================

def draw_board():

    canvas.delete("all")

    for r in range(ROWS):
        for c in range(COLS):

            x1 = c * CELL_SIZE
            y1 = r * CELL_SIZE

            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            piece = game.board[r][c]

            color = "white"

            if piece == PLAYER:
                color = "red"

            elif piece == AI:
                color = "yellow"

            canvas.create_oval(
                x1 + 5,
                y1 + 5,
                x2 - 5,
                y2 - 5,
                fill=color
            )


# =====================================
# GET VALID MOVES
# =====================================

def get_valid_moves():

    return [
        c for c in range(COLS)
        if game.is_valid_move(c)
    ]


# =====================================
# AI MOVE
# =====================================

def ai_move():

    valid_moves = get_valid_moves()

    if len(valid_moves) == 0:
        status_label.config(text="Draw!")
        return

    board = np.array(
        game.board,
        dtype=np.float32
    )

    board = board * AI

    own_pieces = (board == 1).astype(np.float32)
    opp_pieces = (board == -1).astype(np.float32)

    state = np.stack(
        [own_pieces, opp_pieces],
        axis=0
    )

    state_tensor = torch.tensor(
        state,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state_tensor)

    q_values = q_values.squeeze().cpu().numpy()

    masked_q = np.full_like(
        q_values,
        -np.inf
    )

    masked_q[valid_moves] = q_values[valid_moves]

    action = int(np.argmax(masked_q))

    game.current_player = AI
    game.move(action)

    draw_board()

    if game.win_check(AI):
        status_label.config(text="AI Wins!")
        canvas.unbind("<Button-1>")
        return

    if game.is_draw():
        status_label.config(text="Draw!")
        canvas.unbind("<Button-1>")
        return

    status_label.config(text="Your Turn")


# =====================================
# PLAYER CLICK
# =====================================

def handle_click(event):

    col = event.x // CELL_SIZE

    if not game.is_valid_move(col):
        return

    game.current_player = PLAYER
    game.move(col)

    draw_board()

    # =========================
    # CHECK PLAYER WIN
    # =========================

    if game.win_check(PLAYER):
        status_label.config(text="You Win!")
        canvas.unbind("<Button-1>")
        return

    if game.is_draw():
        status_label.config(text="Draw!")
        canvas.unbind("<Button-1>")
        return

    status_label.config(text="AI Thinking...")

    # Small delay so UI updates first
    window.after(300, ai_move)


# =====================================
# RESTART BUTTON
# =====================================

def restart_game():

    global game

    game = Connect4(ROWS, COLS)

    draw_board()

    status_label.config(text="Your Turn")

    canvas.bind("<Button-1>", handle_click)


restart_button = tk.Button(
    window,
    text="Restart",
    font=("Arial", 14),
    command=restart_game
)

restart_button.pack(pady=10)


# =====================================
# START
# =====================================

canvas.bind("<Button-1>", handle_click)

draw_board()

window.mainloop()
