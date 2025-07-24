import math
import numpy as np
import neat
import random
import os
import pickle
import time
import datetime
from neat.reporting import BaseReporter

# Game constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
MARGIN = 2
SCREEN_WIDTH = BOARD_WIDTH * (CELL_SIZE + MARGIN)
SCREEN_HEIGHT = BOARD_HEIGHT * (CELL_SIZE + MARGIN)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)

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
    'T': (128, 0, 128),
    'S': (0, 255, 0),
    'Z': (255, 0, 0),
    'J': (0, 0, 255),
    'L': (255, 165, 0)
}

# Reporter for tracking time
class CompletionTimeReporter(BaseReporter):
    def __init__(self, num_generations):
        super().__init__()
        self.num_generations = num_generations
        self.generation_start_time = None
        self.generation_times = []
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        if self.generation_start_time is not None:
            elapsed = time.time() - self.generation_start_time
            self.generation_times.append(elapsed)
            self.generation_times = self.generation_times[-10:]
            if (self.current_generation % 5 == 0):
                print("Saving Genome...")
            if self.generation_times:
                avg = sum(self.generation_times) / len(self.generation_times)
                remaining = self.num_generations - self.current_generation
                eta = datetime.timedelta(seconds=int(avg * remaining))
                elapsed_total = datetime.timedelta(seconds=int(sum(self.generation_times)))
                print(f"Generation {self.current_generation}/{self.num_generations} complete. "
                      f"Elapsed: {elapsed_total} | ETA: {eta}")

# Seven-bag randomizer
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
    return clear_lines(board)

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
                if success == -1:
                    continue
                features = get_features(test_board)
                score = net.activate(features)[0]
                if score > best_score:
                    best_score = score
                    best_action = (rotation, col)
    return best_action

def play_game(net, max_steps=math.inf):
    board = create_board()
    score = 0
    steps = 0
    bag = SevenBag()
    while steps < max_steps:
        piece = bag.next()
        action = get_best_move(board, piece, net)
        if action is None:
            break
        shape, col = action
        success = place_piece(board, shape, col, piece)
        if success == -1:
            break
        score += 1
        score += pow(success, 4)*10
        steps += 1
    return score

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        score = play_game(net, 1000)
        genome.fitness = score

def run_neat(config_path, gens):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(CompletionTimeReporter(gens))

    def eval_save(genomes, config):
        eval_genomes(genomes, config)
        if (p.generation % 5 == 0):
            best = max(genomes, key=lambda g: g[1].fitness)[1]
            save_genome(best)

    winner = p.run(eval_save, gens)

    with open("best_tetris_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("✅ Best Tetris genome saved.")
    
def save_genome(genome):
    with open("curr_tetris_genome.pkl", "wb") as f:
        pickle.dump(genome, f)

if __name__ == "__main__":
    print("Starting...", flush=True)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_neat(config_path, 200)
