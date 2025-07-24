import math
import numpy as np
import pickle
import pygame
import os
import neat
from collections import Counter


# Constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
MARGIN = 2
SCREEN_WIDTH = BOARD_WIDTH * (CELL_SIZE + MARGIN)
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
        return -1
    for r in range(shape.shape[0]):
        for c in range(shape.shape[1]):
            if shape[r][c]:
                board[row + r][col + c] = piece
    return 0

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
                success = place_piece(test_board, shape, col, piece)
                lines = clear_lines(board)
                if success == -1:
                    continue
                features = get_features(test_board)
                score = net.activate(features)[0]
                if score > best_score:
                    best_score = score
                    best_action = (rotation, col)
    return best_action

def draw_board(screen, board):
    screen.fill(BLACK)
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            cell = board[r][c]
            color = GREY if cell == 0 else TETROMINO_COLORS.get(cell, GREY)
            pygame.draw.rect(
                screen,
                color,
                (c * (CELL_SIZE + MARGIN), r * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE)
            )
    pygame.display.flip()

def visualize_game(net, delay=100, max_steps=math.inf):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Tetris NEAT AI (Best Genome)")
    board = create_board()
    score = 0
    steps = 0
    totalLines = Counter()
    bag = SevenBag()
    running = True

    while running:
        piece = bag.next()
        action = get_best_move(board, piece, net)
        if action is None:
            break
        shape, col = action
        success = place_piece(board, shape, col, piece)
        draw_board(screen, board)
        lines = clear_lines(board)
        if success == -1:
            break
        # if (lines > 0):
        #     pygame.time.delay(lines*2*delay)
        if (lines != 0):
            totalLines[lines] += 1
        score += 1
        score += pow(lines, 4)*10
        steps += 1
        draw_board(screen, board)
        pygame.time.delay(delay)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if steps > max_steps:
            running = False

    print(f"🎮 Game Over! Final score: {score} Steps: {steps} Lines: {totalLines}")
    pygame.quit()

if __name__ == "__main__":
    
    with open("curr_tetris_genome.pkl", "rb") as f:
        genome = pickle.load(f)

    config_path = os.path.join(os.path.dirname(__file__), "neat-config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for i in range(10):
        visualize_game(net, delay=0)
