import math
import numpy as np
import pickle
import pygame
import os
import neat
from collections import Counter
import random
import sys

# ---------------------------
# Configuration / constants
# ---------------------------
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
MARGIN = 2
STATS_WIDTH = 300
SCREEN_WIDTH = BOARD_WIDTH * (CELL_SIZE + MARGIN) + STATS_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT * (CELL_SIZE + MARGIN)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)

PIECE_ORDER = ['I', 'O', 'T', 'S', 'Z', 'J', 'L']  # must match one-hot encoding in training

# Node names mapping for potential visualization (35 inputs + 1 output)
node_names = {
    -1: "height_0",
    -2: "height_1",
    -3: "height_2",
    -4: "height_3",
    -5: "height_4",
    -6: "height_5",
    -7: "height_6",
    -8: "height_7",
    -9: "height_8",
    -10: "height_9",
    -11: "holes",
    -12: "bumpiness",
    -13: "avg_height",
    -14: "cur_I",
    -15: "cur_O",
    -16: "cur_T",
    -17: "cur_S",
    -18: "cur_Z",
    -19: "cur_J",
    -20: "cur_L",
    -21: "next_I",
    -22: "next_O",
    -23: "next_T",
    -24: "next_S",
    -25: "next_Z",
    -26: "next_J",
    -27: "next_L",
    -28: "held_I",
    -29: "held_O",
    -30: "held_T",
    -31: "held_S",
    -32: "held_Z",
    -33: "held_J",
    -34: "held_L",
    -35: "hold_locked",
    0: "move_score"
}

# Tetromino definitions
TETROMINOES = {
    'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
    'O': [[[1, 1], [1, 1]]],
    'T': [[[0, 1, 0], [1, 1, 1]],
          [[1, 0], [1, 1], [1, 0]],
          [[1, 1, 1], [0, 1, 0]],
          [[0, 1], [1, 1], [0, 1]]],
    'S': [[[0, 1, 1], [1, 1, 0]],
          [[1, 0], [1, 1], [0, 1]]],
    'Z': [[[1, 1, 0], [0, 1, 1]],
          [[0, 1], [1, 1], [1, 0]]],
    'J': [[[1, 0, 0], [1, 1, 1]],
          [[1, 1], [1, 0], [1, 0]],
          [[1, 1, 1], [0, 0, 1]],
          [[0, 1], [0, 1], [1, 1]]],
    'L': [[[0, 0, 1], [1, 1, 1]],
          [[1, 0], [1, 0], [1, 1]],
          [[1, 1, 1], [1, 0, 0]],
          [[1, 1], [0, 1], [0, 1]]]
}

TETROMINO_COLORS = {
    'I': (0, 255, 255),
    'O': (255, 255, 0),
    'T': (255, 0, 255),
    'S': (0, 255, 0),
    'Z': (255, 0, 0),
    'J': (0, 0, 255),
    'L': (255, 165, 0)
}

# ---------------------------
# Utility / board functions
# ---------------------------
class SevenBag:
    def __init__(self):
        self.bag = []
        self.refill_bag()

    def refill_bag(self):
        self.bag = list(TETROMINOES.keys())
        random.shuffle(self.bag)

    def next(self):
        if not self.bag:
            self.refill_bag()
        return self.bag.pop()

def create_board():
    return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=object)

def check_collision(board, shape, row, col):
    shape = np.array(shape)
    if row + shape.shape[0] > BOARD_HEIGHT or col < 0 or col + shape.shape[1] > BOARD_WIDTH:
        return True
    return np.any((shape == 1) & (board[row:row + shape.shape[0], col:col + shape.shape[1]] != 0))

def place_piece(board, shape, col, piece):
    """Place shape into board at column col; returns (lines_cleared, placed_row) or (-1,-1) on loss"""
    shape = np.array(shape)
    for row in range(BOARD_HEIGHT):
        if check_collision(board, shape, row, col):
            row -= 1
            break
    else:
        row = BOARD_HEIGHT - shape.shape[0]
    if row < 0:
        return -1, -1
    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r][c]:
                board[row + r][col + c] = piece
    return clear_lines(board), row

def clear_lines(board):
    full_rows = np.where(np.all(board != 0, axis=1))[0]
    for r in full_rows:
        if r > 0:
            board[1:r+1] = board[0:r]
        board[0] = 0
    return len(full_rows)

def one_hot_piece(piece):
    """Return one-hot (length 7) encoding for a piece name; returns zeros if None."""
    vec = [0] * len(PIECE_ORDER)
    if piece is None:
        return vec
    vec[PIECE_ORDER.index(piece)] = 1
    return vec

def get_features(board):
    heights = [
        next((BOARD_HEIGHT - r for r in range(BOARD_HEIGHT) if board[r][c]), 0)
        for c in range(BOARD_WIDTH)
    ]
    holes = sum(
        1
        for c in range(BOARD_WIDTH)
        for r in range(BOARD_HEIGHT)
        if board[r][c] == 0 and any(board[r2][c] != 0 for r2 in range(r))
    )
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    avg_height = sum(heights) / len(heights)
    return heights + [holes, bumpiness, avg_height]

# ---------------------------
# Move search / evaluation
# ---------------------------
def get_best_move(board, current_piece, next_piece, held_piece, hold_locked, net):
    """
    Search possible placements for:
      - placing current_piece
      - if not hold_locked: using HOLD then placing the swapped piece (either held_piece or next_piece)
    Return action dict:
      {'type': 'place' or 'hold_place',
       'rotation': rotation_shape,
       'col': column,
       'placed_piece': piece_placed,
       'resulting_held': piece_in_hold_after_action}
    """
    best_score = -math.inf
    best_action = None

    # 1) Place current_piece normally
    for rotation in TETROMINOES[current_piece]:
        shape = np.array(rotation)
        for col in range(BOARD_WIDTH - shape.shape[1] + 1):
            test_board = board.copy()
            if not check_collision(test_board, shape, 0, col):
                lines, row = place_piece(test_board, shape, col, current_piece)
                if lines == -1:
                    continue
                features = get_features(test_board)
                features_ext = (
                    features
                    + one_hot_piece(current_piece)
                    + one_hot_piece(next_piece)
                    + one_hot_piece(held_piece)
                    + [1 if hold_locked else 0]
                )
                score = net.activate(features_ext)[0]
                if score > best_score:
                    best_score = score
                    best_action = {
                        'type': 'place',
                        'rotation': rotation,
                        'col': col,
                        'placed_piece': current_piece,
                        'resulting_held': held_piece
                    }

    # 2) Consider hold+place (if allowed)
    if not hold_locked:
        # If there is a held piece, after holding you place that piece.
        # If no held piece, after holding you typically place the next piece.
        swap_target = held_piece if held_piece is not None else next_piece
        if swap_target is not None:
            for rotation in TETROMINOES[swap_target]:
                shape = np.array(rotation)
                for col in range(BOARD_WIDTH - shape.shape[1] + 1):
                    test_board = board.copy()
                    if not check_collision(test_board, shape, 0, col):
                        lines, row = place_piece(test_board, shape, col, swap_target)
                        if lines == -1:
                            continue
                        features = get_features(test_board)
                        # After holding, resulting_held becomes current_piece.
                        # For the network input, we represent the situation after the hold+place:
                        # - current (the piece just placed) = swap_target
                        # - next stays as next_piece (we keep it for consistency)
                        # - held becomes current_piece (since we swapped)
                        # - hold_locked will be true after a hold+place
                        features_ext = (
                            features
                            + one_hot_piece(swap_target)
                            + one_hot_piece(next_piece)
                            + one_hot_piece(current_piece)
                            + [1]  # hold_locked true after hold+place
                        )
                        score = net.activate(features_ext)[0]
                        if score > best_score:
                            best_score = score
                            best_action = {
                                'type': 'hold_place',
                                'rotation': rotation,
                                'col': col,
                                'placed_piece': swap_target,
                                'resulting_held': current_piece
                            }

    return best_action

# ---------------------------
# Drawing / visualization
# ---------------------------
def draw_board(screen, board):
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            cell = board[r][c]
            color = GREY if cell == 0 else TETROMINO_COLORS.get(cell, GREY)
            pygame.draw.rect(
                screen,
                color,
                (c * (CELL_SIZE + MARGIN), r * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE)
            )

def draw_stats(screen, font, score, steps, total_lines, current_piece, next_piece, held_piece, hold_locked):
    base_x = BOARD_WIDTH * (CELL_SIZE + MARGIN) + 10
    lines = [
        f"Score: {score}",
        f"Pieces: {steps}",
        f"Level: {sum(k * v for k, v in total_lines.items())//10 + 1}",
        f"Lines: {sum(k * v for k, v in total_lines.items())}",
        "",
        f"Current: {current_piece}",
        f"Next: {next_piece}",
        f"Held: {held_piece}",
        f"Hold Locked: {hold_locked}",
        "",
        "Breakdown:"
    ] + [f"{k}L: {v}" for k, v in sorted(total_lines.items())]

    for i, line in enumerate(lines):
        text = font.render(line, True, WHITE)
        screen.blit(text, (base_x, 20 + i * 22))

# ---------------------------
# Visualize / run one game
# ---------------------------
def visualize_game(net, delay=100, max_steps=math.inf):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris NEAT AI (Best Genome)")
    font = pygame.font.SysFont("consolas", 18)

    board = create_board()
    score = 0
    steps = 0
    total_lines = Counter()
    bag = SevenBag()
    running = True

    # initialize current and next
    current = bag.next()
    next_piece = bag.next()
    held = None
    hold_locked = False

    while running:
        level = sum(k * v for k, v in total_lines.items())//10 + 1

        action = get_best_move(board, current, next_piece, held, hold_locked, net)
        if action is None:
            break

        # Execute action
        if action['type'] == 'place':
            placed_piece = action['placed_piece']  # should be current
            lines, row = place_piece(board, action['rotation'], action['col'], placed_piece)
            if lines == -1:
                break
            # scoring heuristics (kept similar to your previous system)
            score += (BOARD_HEIGHT - row) * 2
            if lines == 1:
                score += 100 * level
            elif lines == 2:
                score += 300 * level
            elif lines == 3:
                score += 500 * level
            elif lines == 4:
                score += 800 * level
            if lines != 0:
                total_lines[lines] += 1

            # advance bag: current becomes next, next pulled from bag
            current = next_piece
            next_piece = bag.next()
            # held remains as action['resulting_held'] (usually unchanged)
            held = action['resulting_held']
            # after a placement, you can hold again
            hold_locked = False
            steps += 1

        elif action['type'] == 'hold_place':
            placed_piece = action['placed_piece']  # either held_piece or next_piece
            lines, row = place_piece(board, action['rotation'], action['col'], placed_piece)
            if lines == -1:
                break
            score += (BOARD_HEIGHT - row) * 2
            if lines == 1:
                score += 100 * level
            elif lines == 2:
                score += 300 * level
            elif lines == 3:
                score += 500 * level
            elif lines == 4:
                score += 800 * level
            if lines != 0:
                total_lines[lines] += 1

            # Update hold and queue semantics:
            new_held = action['resulting_held']  # should be the pre-swap current
            if held is None:
                # If held was empty, the action placed next_piece -> we should advance queue twice:
                # We already used next_piece as the placed piece, so current becomes a fresh next
                current = bag.next()
                next_piece = bag.next()
            else:
                # If there was a held piece, we placed held_piece; current becomes next_piece
                current = next_piece
                next_piece = bag.next()
            held = new_held
            # After hold+place, holds are locked until next placement
            hold_locked = True
            steps += 1

        else:
            break

        # Draw
        screen.fill(BLACK)
        draw_board(screen, board)
        draw_stats(screen, font, score, steps, total_lines, current, next_piece, held, hold_locked)
        pygame.display.flip()

        # event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.time.delay(delay)

        if steps > max_steps:
            running = False

    print(f"\nGame Over! Final score: {score} Steps: {steps} Lines: {total_lines}")
    pygame.quit()

# ---------------------------
# Main: load genome, run visualizer
# ---------------------------
if __name__ == "__main__":
    # load genome
    genome = None
    try:
        with open("curr_tetris_genome.pkl", "rb") as f:
            genome = pickle.load(f)
    except Exception as e:
        print("Couldn't load best_tetris_genome.pkl:", e)
        sys.exit(1)

    # load neat config
    config_path = os.path.join(os.path.dirname(__file__), "neat-config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Run a few visualized games
    for i in range(5):
        visualize_game(net, delay=500, max_steps=1000)
