import math
import numpy as np
import neat
import random
import os
import pickle
from multiprocessing import Pool, cpu_count
import threading

import pygame

BOARD_WIDTH = 10
BOARD_HEIGHT = 20

CELL_SIZE = 30
MARGIN = 2
SCREEN_WIDTH = BOARD_WIDTH * (CELL_SIZE + MARGIN)
SCREEN_HEIGHT = BOARD_HEIGHT * (CELL_SIZE + MARGIN)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)
BLUE = (0, 100, 255)

TETROMINOES = {
    'I': [[[1, 1, 1, 1]],
          [[1], [1], [1], [1]]],
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

import time
import datetime
from neat.reporting import BaseReporter

class CompletionTimeReporter(BaseReporter):
    def __init__(self, num_generations):
        super().__init__()
        self.num_generations = num_generations
        self.generation_start_time = None
        self.generation_times = []
        self.current_generation = 0

    def start_generation(self, generation):
        self.current_generation = generation  # NEAT passes this in correctly
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):  # MUST match NEAT's signature
        if self.generation_start_time is not None:
            elapsed = time.time() - self.generation_start_time
            self.generation_times.append(elapsed)

            # Limit how many generations to consider for averaging
            self.generation_times = self.generation_times[-10:]

            if self.generation_times:
                average_generation_time = sum(self.generation_times) / len(self.generation_times)
                generations_remaining = self.num_generations - self.current_generation
                estimated_remaining_time_seconds = average_generation_time * generations_remaining
                eta = datetime.timedelta(seconds=int(estimated_remaining_time_seconds))
                elapsed = datetime.timedelta(seconds=int(sum(self.generation_times)))

                print(f"Generation {self.current_generation}/{self.num_generations} complete. "
                      f"Elapsed: {elapsed} | ETA: {eta}")


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
    return np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)

def check_collision(board, shape, row, col):
    shape = np.array(shape)
    if row + shape.shape[0] > BOARD_HEIGHT or col < 0 or col + shape.shape[1] > BOARD_WIDTH:
        return True
    return np.any((shape == 1) & (board[row:row + shape.shape[0], col:col + shape.shape[1]] == 1))

def place_piece(board, shape, col):
    shape = np.array(shape)
    for row in range(BOARD_HEIGHT):
        if check_collision(board, shape, row, col):
            row -= 1
            break
    else:
        row = BOARD_HEIGHT - shape.shape[0]
    if row < 0:
        return -1  # Game over
    board[row:row + shape.shape[0], col:col + shape.shape[1]] += shape
    return clear_lines(board)

def clear_lines(board):
    full_rows = np.where(np.all(board == 1, axis=1))[0]
    for r in full_rows:
        board[1:r+1] = board[0:r]
        board[0] = 0
    return len(full_rows)

def get_features(board):
    # Features: column heights, holes, bumpiness, avg height
    heights = [next((BOARD_HEIGHT - r for r in range(BOARD_HEIGHT) if board[r][c]), 0) for c in range(BOARD_WIDTH)]
    holes = sum((1 for c in range(BOARD_WIDTH)
                 for r in range(BOARD_HEIGHT)
                 if board[r][c] == 0 and any(board[r2][c] == 1 for r2 in range(r))))
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
                success = place_piece(test_board, shape, col)
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
            break  # Game over
        shape, col = action
        success = place_piece(board, shape, col)
        if success == -1:
            break # Game over
        score += 1
        score += pow(success, 4)*10
        steps += 1
    return score

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        score = play_game(net, 500)
        genome.fitness = score

def run_neat(config_path, gens):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    p.add_reporter(CompletionTimeReporter(gens))

    def eval_with_visualization(genomes, config):
        eval_genomes(genomes, config)
        # After each generation, visualize the best genome
        best = max(genomes, key=lambda g: g[1].fitness)[1]
        best_net = neat.nn.FeedForwardNetwork.create(best, config)
        thread = threading.Thread(target=visualize_game, args=(best_net, 1, False, 200), daemon=True)
        thread.start()

    winner = p.run(eval_with_visualization, gens)

    with open("best_tetris_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("✅ Best Tetris genome saved.")

    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    visualize_game(best_net, end=True)

def draw_board(screen, board):
    screen.fill(BLACK)
    for r in range(BOARD_HEIGHT):
        for c in range(BOARD_WIDTH):
            color = GREY if board[r][c] == 0 else BLUE
            pygame.draw.rect(
                screen,
                color,
                (c * (CELL_SIZE + MARGIN), r * (CELL_SIZE + MARGIN), CELL_SIZE, CELL_SIZE)
            )
    pygame.display.flip()

def visualize_game(net, delay=100, end=False, max_steps = math.inf):
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NEAT Tetris AI")

    board = create_board()
    score = 0
    clock = pygame.time.Clock()
    steps = 0
    
    running = True
    bag = SevenBag()
    while running:
        piece = bag.next()
        action = get_best_move(board, piece, net)
        if action is None:
            running = False
            continue

        shape, col = action
        success = place_piece(board, shape, col)
        if success == -1:
            running = False
            continue

        score += 1
        score += pow(success, 4)*10
        steps += 1

        draw_board(screen, board)
        pygame.time.delay(delay)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if (steps > max_steps):
            running = False
        with open("best_tetris_genome.pkl", "wb") as f:
            pickle.dump(net, f)
    if (end):
        print("🎮 Game Over! Final score:", score)
    pygame.quit()

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "neat-config.txt")
    run_neat(config_path, 100)
