import math
import numpy as np
import neat
import random
import os
import pickle
import time
import datetime
from multiprocessing import Pool, cpu_count
from neat.reporting import BaseReporter
import sys

sys.stdout = open('out.txt', 'a', buffering=1)

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

# Tetromino definitions (unchanged)
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

# Reporter for tracking time (unchanged)
class CompletionTimeReporter(BaseReporter):
    def __init__(self, num_generations):
        super().__init__()
        self.num_generations = num_generations
        self.generation_start_time = None
        self.generation_times = []
        self.allgeneration_times = []
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        if self.generation_start_time is not None:
            elapsed = time.time() - self.generation_start_time
            self.allgeneration_times.append(elapsed)
            self.generation_times.append(elapsed)
            self.generation_times = self.generation_times[-10:]
            if self.generation_times:
                avg = sum(self.generation_times) / len(self.generation_times)
                remaining = self.num_generations - self.current_generation
                eta = datetime.timedelta(seconds=int(avg * remaining))
                elapsed_total = datetime.timedelta(seconds=int(sum(self.allgeneration_times)))
                print(f"Generation {self.current_generation}/{self.num_generations} complete. "
                      f"Elapsed: {elapsed_total} | ETA: {eta}")

# Seven-bag randomizer (unchanged)
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

# Helpers for encoding pieces into inputs (order must be consistent with config)
PIECE_ORDER = list(TETROMINOES.keys())  # ['I','O','T','S','Z','J','L']

def one_hot_piece(piece):
    """Return one-hot list (length 7) for a piece name, or all zeros if None."""
    vec = [0]*len(PIECE_ORDER)
    if piece is None:
        return vec
    idx = PIECE_ORDER.index(piece)
    vec[idx] = 1
    return vec

# Extended get_best_move: considers holding and seeing the next piece.
def get_best_move(board, current_piece, next_piece, held_piece, hold_locked, net):
    """
    Evaluate:
     - placing current_piece now (all rotations & columns)
     - if hold allowed (not hold_locked): hold current_piece, then place held_piece (if exists) or next_piece
       (this simulates the standard Tetris swap/hold mechanic).
    Returns
      best_action dict with keys:
        - type: 'place' or 'hold_place'
        - rotation: rotation shape chosen
        - col: column chosen
        - resulting_held: piece that will be in hold after action
    """
    best_score = -math.inf
    best = None

    # 1) Consider placing current_piece now
    for rotation in TETROMINOES[current_piece]:
        shape = np.array(rotation)
        for col in range(BOARD_WIDTH - shape.shape[1] + 1):
            test_board = board.copy()
            if not check_collision(test_board, shape, 0, col):
                success = place_piece(test_board, shape, col, current_piece)
                if success == -1:
                    continue
                features = get_features(test_board)
                # Build extended feature vector:
                features_ext = features + one_hot_piece(current_piece) + one_hot_piece(next_piece) + one_hot_piece(held_piece) + [1 if hold_locked else 0]
                score = net.activate(features_ext)[0]
                if score > best_score:
                    best_score = score
                    best = {'type': 'place', 'rotation': rotation, 'col': col, 'resulting_held': held_piece, 'placed_piece': current_piece}

    # 2) Consider using hold (if allowed)
    if not hold_locked:
        # If held_piece exists, after hold you'll place the held_piece.
        # If no held_piece, after hold you'll place next_piece (and you'll pull a fresh next later).
        swap_target = held_piece if held_piece is not None else next_piece
        # If there's no piece to place after swap (shouldn't happen), skip.
        if swap_target is not None:
            for rotation in TETROMINOES[swap_target]:
                shape = np.array(rotation)
                for col in range(BOARD_WIDTH - shape.shape[1] + 1):
                    test_board = board.copy()
                    if not check_collision(test_board, shape, 0, col):
                        success = place_piece(test_board, shape, col, swap_target)
                        if success == -1:
                            continue
                        features = get_features(test_board)
                        # resulting_held after doing this hold action becomes current_piece
                        features_ext = features + one_hot_piece(swap_target) + one_hot_piece(next_piece) + one_hot_piece(current_piece) + [1]  # hold_locked will be set true
                        score = net.activate(features_ext)[0]
                        if score > best_score:
                            best_score = score
                            best = {'type': 'hold_place', 'rotation': rotation, 'col': col, 'resulting_held': current_piece, 'placed_piece': swap_target}

    return best

def play_game(net, max_steps=5000):
    board = create_board()
    score = 0
    steps = 0
    bag = SevenBag()
    # initialize current and next
    current = bag.next()
    next_piece = bag.next()
    held = None
    hold_locked = False  # prevents holding again until a placement happens

    while steps < max_steps:
        action = get_best_move(board, current, next_piece, held, hold_locked, net)
        if action is None:
            break

        if action['type'] == 'place':
            # place current piece (action['placed_piece'] should equal current)
            placed_piece = action['placed_piece']
            success = place_piece(board, action['rotation'], action['col'], placed_piece)
            if success == -1:
                break
            score += 0.1
            if (success > 1):
            	score += pow(success, 5)
            # move to next piece from queue
            current = next_piece
            next_piece = bag.next()
            held = action['resulting_held']  # unchanged usually
            hold_locked = False  # after placing, holding becomes available
            steps += 1

        elif action['type'] == 'hold_place':
            # perform hold swap: resulting_held is the piece that will now be in hold
            # we place action['placed_piece'] (which was either held or next_piece)
            placed_piece = action['placed_piece']
            success = place_piece(board, action['rotation'], action['col'], placed_piece)
            if success == -1:
                break
            score += 1
            score += pow(success, 4) * 10
            # update hold and current/next according to standard swap rules:
            # resulting_held was set to the pre-swap current piece
            new_held = action['resulting_held']
            if held is None:
                # swapping into an empty hold: current becomes next_piece, and we pulled a new next already when evaluating
                # We used swap_target = next_piece earlier; after we place it, current should become the subsequent piece (we already pulled one when game runs)
                # To simulate correct queue behavior:
                current = bag.next()
                # next_piece already advanced implicitly (we consumed next_piece), but to be safe we set next_piece to bag.next()
                next_piece = bag.next()
            else:
                # swapping with held: current becomes held (we placed held), next stays the same
                current = next_piece
                next_piece = bag.next()
            held = new_held
            # after a hold+place, you cannot hold again until you place the next piece
            hold_locked = True
            steps += 1

        else:
            # unknown action
            break

    return score

# Multiprocessing version of genome evaluation (unchanged aside from function signatures)
def eval_genome(args):
    genome_id, genome, config = args
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = [play_game(net, 5000) for _ in range(5)]
    genome.fitness = sum(scores) / len(scores)
    return genome_id, genome

def eval_genomes(genomes, config):
    cpus = max(1, cpu_count()//2)

    with Pool(cpus) as pool:
        args = [(genome_id, genome, config) for genome_id, genome in genomes]
        results = pool.map(eval_genome, args)

    id_to_genome = dict(genomes)
    for genome_id, genome in results:
        id_to_genome[genome_id].fitness = genome.fitness

def run_neat(config_path, gens):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # If you have an existing genome, you can still load it
    try:
        with open("best_tetris_genome.pkl", "rb") as f:
            genome = pickle.load(f)
    except Exception:
        genome = None

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(CompletionTimeReporter(num_generations=gens))

    def eval_save(genomes, config):
        eval_genomes(genomes, config)
        if (p.generation % 5 == 0):
            print("Saving Genome...")
            best = max(genomes, key=lambda g: g[1].fitness)[1]
            save_genome(best)
        sys.stdout.flush()

    winner = p.run(eval_save, gens)

    with open("best_tetris_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("Best Tetris genome saved.")

def save_genome(genome):
    with open("curr_tetris_genome.pkl", "wb") as f:
        pickle.dump(genome, f)

if __name__ == "__main__":
    sys.stdout.flush()
    print(f"Starting... CPUS: {cpu_count()//2} / {cpu_count()}", flush=True)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_neat(config_path, 1000)
    sys.stdout.close()
