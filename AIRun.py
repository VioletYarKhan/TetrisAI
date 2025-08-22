import math
import numpy as np
import pickle
import pygame
import os
import neat
import visualize  # ensure this is in your directory
from collections import Counter


level = 1
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
    0: "move_score"
}

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

TETROMINOES = {
    'I': [[[1, 1, 1, 1]], [[1], [1], [1], [1]]],
    'O': [[[1, 1], [1, 1]]],
    'T': [[[0, 1, 0], [1, 1, 1]], [[1, 0], [1, 1], [1, 0]], [[1, 1, 1], [0, 1, 0]], [[0, 1], [1, 1], [0, 1]]],
    'S': [[[0, 1, 1], [1, 1, 0]], [[1, 0], [1, 1], [0, 1]]],
    'Z': [[[1, 1, 0], [0, 1, 1]], [[0, 1], [1, 1], [1, 0]]],
    'J': [[[1, 0, 0], [1, 1, 1]], [[1, 1], [1, 0], [1, 0]], [[1, 1, 1], [0, 0, 1]], [[0, 1], [0, 1], [1, 1]]],
    'L': [[[0, 0, 1], [1, 1, 1]], [[1, 0], [1, 0], [1, 1]], [[1, 1, 1], [1, 0, 0]], [[1, 1], [0, 1], [0, 1]]]
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

class SevenBag:
    def __init__(self):
        self.bag = []
        self.refill_bag()
    def refill_bag(self):
        self.bag = list(TETROMINOES.keys())
        np.random.shuffle(self.bag)
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

def get_features(board):
    heights = [next((BOARD_HEIGHT - r for r in range(BOARD_HEIGHT) if board[r][c]), 0) for c in range(BOARD_WIDTH)]
    holes = sum((1 for c in range(BOARD_WIDTH)
                 for r in range(BOARD_HEIGHT)
                 if board[r][c] == 0 and any(board[r2][c] != 0 for r2 in range(r))))
    bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
    avg_height = sum(heights)/len(heights)
    return heights + [holes, bumpiness, avg_height]

def get_best_move(board, piece, net):
    best_score = -math.inf
    best_action = None
    for rotation in TETROMINOES[piece]:
        shape = np.array(rotation)
        for col in range(BOARD_WIDTH - shape.shape[1] + 1):
            test_board = board.copy()
            if not check_collision(test_board, shape, 0, col):
                success, row = place_piece(test_board, shape, col, piece)
                if success == -1:
                    continue
                features = get_features(test_board)
                score = net.activate(features)[0]
                if score > best_score:
                    best_score = score
                    best_action = (rotation, col)
    return best_action

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

def draw_stats(screen, font, score, steps, total_lines):
    base_x = BOARD_WIDTH * (CELL_SIZE + MARGIN) + 10
    lines = [
        f"Score: {score}",
        f"Pieces: {steps}",
        f"Level: {sum(k * v for k, v in total_lines.items())//10 + 1}",
        f"Lines: {sum(k * v for k, v in total_lines.items())}",
        "Breakdown:" 
    ] + [f"{k}L: {v}" for k, v in sorted(total_lines.items())]

    for i, line in enumerate(lines):
        text = font.render(line, True, WHITE)
        screen.blit(text, (base_x, 20 + i * 25))

def visualize_game(net, delay=100, max_steps=math.inf):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris NEAT AI (Best Genome) + NN Visualization")
    font = pygame.font.SysFont("consolas", 20)

    board = create_board()
    score = 0
    steps = 0
    total_lines = Counter()
    bag = SevenBag()
    running = True

    while running:
        level = sum(k * v for k, v in total_lines.items())//10 + 1
        piece = bag.next()
        action = get_best_move(board, piece, net)
        if action is None:
            break
        shape, col = action
        success, row = place_piece(board, shape, col, piece)
        if success == -1:
            break
        lines = success
        score += (BOARD_HEIGHT-row)*2
        if (success == 1):
            success += 100*level
        elif (success == 2):
            success += 300*level
        elif (success == 3):
            success += 500*level
        elif (success == 4):
            success += 800*level
        if lines != 0:
            total_lines[lines] += 1
        steps += 1

        screen.fill(BLACK)
        draw_board(screen, board)
        draw_stats(screen, font, score, steps, total_lines)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.time.delay(delay)

        if steps > max_steps:
            running = False

    print(f"\n🎮 Game Over! Final score: {score} Steps: {steps} Lines: {total_lines}")
    pygame.quit()

if __name__ == "__main__":
    with open("best_tetris_genome.pkl", "rb") as f:
        genome = pickle.load(f)

    config_path = os.path.join(os.path.dirname(__file__), "neat-config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    visualize.draw_net(config, genome, view=True, node_names=node_names)

    for i in range(5):
        visualize_game(net, delay=10)
