import pygame
import random
import time
from collections import deque
from dataclasses import dataclass, field
import heapq

GRID_SIZE = 10
CELL_SIZE = 60
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
MAX_APPLES = 35
FPS = 2500

WHITE = (245, 245, 245)
BLACK = (25, 25, 25)
RED = (220, 20, 60)
GREEN = (46, 204, 113)
BLUE = (52, 152, 219)
GRAY = (127, 140, 141)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRS = [UP, DOWN, LEFT, RIGHT]

def add(a, b):
    return (a[0] + b[0], a[1] + b[1])

def in_bounds(p):
    x, y = p
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

@dataclass(order=True)
class Node:
    f: int
    g: int
    pos: tuple = field(compare=False)
    parent: tuple = field(compare=False, default=None)

@dataclass
class AStarResult:
    found: bool
    path: list
    visited: int

def reconstruct_path(came_from, end):
    path = [end]
    cur = end
    while cur in came_from:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

def astar(start, goal, blocked):
    if start == goal:
        return AStarResult(True, [start], 0)

    open_heap = []
    heapq.heappush(open_heap, Node(f=manhattan(start, goal), g=0, pos=start))
    came_from = {}
    g_score = {start: 0}
    visited = 0
    closed = set()

    while open_heap:
        current = heapq.heappop(open_heap)
        visited += 1
        if current.pos == goal:
            path = reconstruct_path(came_from, current.pos)
            return AStarResult(True, path, visited)

        if current.pos in closed:
            continue
        closed.add(current.pos)

        for d in DIRS:
            nxt = add(current.pos, d)
            if not in_bounds(nxt) or nxt in blocked:
                continue
            tentative_g = g_score[current.pos] + 1
            if tentative_g < g_score.get(nxt, 10**9):
                came_from[nxt] = current.pos
                g_score[nxt] = tentative_g
                f = tentative_g + manhattan(nxt, goal)
                heapq.heappush(open_heap, Node(f=f, g=tentative_g, pos=nxt))

    return AStarResult(False, [], visited)

def bfs_path(start, goal, blocked):
    if start == goal:
        return [start]
    q = deque([start])
    visited = {start}
    parent = {}
    while q:
        u = q.popleft()
        for d in DIRS:
            v = add(u, d)
            if not in_bounds(v) or v in blocked or v in visited:
                continue
            parent[v] = u
            if v == goal:
                path = [v]
                cur = v
                while cur != start:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                return path
            visited.add(v)
            q.append(v)
    return []

def simulate_path_and_check_tail(snake_deque, path, apple):
    sim = deque(snake_deque)
    for idx in range(1, len(path)):
        step = path[idx]
        ate = (step == apple)
        sim.append(step)
        if not ate:
            sim.popleft()
        if len(set(sim)) != len(sim):
            return False, None

    n_head = sim[-1]
    n_tail = sim[0]
    blocked = set(sim)
    blocked.discard(n_tail)
    return (len(bfs_path(n_head, n_tail, blocked)) > 0, sim)

def flood_free_space(start_cell, snake_deque):
    occupied = set(snake_deque)
    tail = snake_deque[0]
    occupied.discard(tail)
    q = deque([start_cell])
    seen = {start_cell}
    count = 0
    while q:
        u = q.popleft()
        count += 1
        for d in DIRS:
            v = add(u, d)
            if not in_bounds(v) or v in occupied or v in seen:
                continue
            seen.add(v)
            q.append(v)
    return count

def best_safe_step_following_tail(start, snake_deque):
    tail = snake_deque[0]
    candidates = []
    for d in DIRS:
        nm = add(start, d)
        if not in_bounds(nm):
            continue
        grew = False
        sim = deque(snake_deque)
        sim.append(nm)
        if not grew:
            sim.popleft()
        if len(set(sim)) != len(sim):
            continue
        space = flood_free_space(nm, sim)
        dist_tail = manhattan(nm, tail)
        candidates.append((space, dist_tail, nm))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]

class Snake:
    def __init__(self):
        cx, cy = GRID_SIZE // 2, GRID_SIZE // 2
        self.snake = deque([(cx - 1, cy), (cx, cy), (cx + 1, cy)])
        self.apple = self.spawn_apple()
        self.apples_eaten = 0
        self.game_over = False

    def spawn_apple(self):
        free = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                if (x, y) not in self.snake]
        return random.choice(free) if free else None

    def head(self):
        return self.snake[-1]

    def tail(self):
        return self.snake[0]

    def occupied_set(self):
        return set(self.snake)

    def step(self):
        if self.game_over:
            return

        start = self.head()
        tail_now = self.tail()
        blocked_for_apple = self.occupied_set().copy()
        blocked_for_apple.discard(tail_now)

        next_move = None
        a_res = astar(start, self.apple, blocked_for_apple)
        if a_res.found and len(a_res.path) >= 2:
            path_safe, _ = simulate_path_and_check_tail(self.snake, a_res.path, self.apple)
            if path_safe:
                next_move = a_res.path[1]

        if next_move is None:
            nm = best_safe_step_following_tail(start, self.snake)
            if nm is not None:
                next_move = nm

        if next_move is None:
            candidates = []
            for d in DIRS:
                nm = add(start, d)
                if not in_bounds(nm):
                    continue
                grew = (nm == self.apple)
                next_snake = deque(self.snake)
                next_snake.append(nm)
                if not grew:
                    next_snake.popleft()
                if len(set(next_snake)) == len(next_snake):
                    space = flood_free_space(nm, next_snake)
                    dist_tail = manhattan(nm, next_snake[0])
                    candidates.append((space, dist_tail, nm))
            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
                next_move = candidates[0][2]

        if next_move is None:
            self.game_over = True
            return

        self.snake.append(next_move)
        ate = (next_move == self.apple)
        if ate:
            self.apples_eaten += 1
            if self.apples_eaten >= MAX_APPLES:
                self.game_over = True
            else:
                self.apple = self.spawn_apple()
        else:
            self.snake.popleft()

def draw_grid(screen):
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y), 1)

def draw_cell(screen, p, color):
    x, y = p
    r = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, color, r)

# --------- NUEVO: intento visual con log ---------
def run_single_attempt(render=True):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake - Intento visual")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    game = Snake()
    running = True
    while running:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if not game.game_over:
            game.step()

        if render:
            screen.fill(BLACK)
            draw_grid(screen)
            if game.apple is not None:
                draw_cell(screen, game.apple, RED)
            for i, seg in enumerate(game.snake):
                draw_cell(screen, seg, GREEN if i < len(game.snake) - 1 else BLUE)
            pygame.display.flip()

        if game.game_over:
            pygame.time.wait(800)
            running = False

    apples = game.apples_eaten
    pygame.quit()
    return apples, (apples >= MAX_APPLES)

def run_trials(n=20, render=True):
    successes = 0
    for i in range(1, n + 1):
        apples, ok = run_single_attempt(render=render)
        if ok:
            print(f"logrado intento {i} manzanas {apples}/{MAX_APPLES}")
            successes += 1
        else:
            print(f"no logrado intento {i} manzanas {apples}/{MAX_APPLES}")
    print("—" * 40)
    print(f"Éxitos: {successes}/{n}  ({successes/n*100:.1f} %)  |  Fallos: {n - successes}")

if __name__ == "__main__":
    run_trials(n=10, render=True)