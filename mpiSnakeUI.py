from mpi4py import MPI
import random
import time
import sys
import os
import pygame

# --- Configuration ---
BOARD_SIZE = 10
SNAKE_LENGTH = 5
CELL_SIZE = 40
FPS = 5

def get_valid_moves(head, my_body, other_body):
    moves = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    for dx, dy in directions:
        nx, ny = head[0] + dx, head[1] + dy
        if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if (nx, ny) not in my_body and (nx, ny) not in other_body:
                moves.append((nx, ny))
    return moves

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 3:
        if rank == 0:
            print("Run with: mpiexec -n 3 python mpi_snake_pygame.py")
        sys.exit()

    # --- SCATTER roles ---
    if rank == 0:
        roles = ['master', 's1', 's2']
    else:
        roles = None

    my_role = comm.scatter(roles, root=0)

    # --- INIT MASTER ---
    if rank == 0:
        pygame.init()
        WIDTH = BOARD_SIZE * CELL_SIZE
        HEIGHT = BOARD_SIZE * CELL_SIZE

        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("MPI Snake")
        clock = pygame.time.Clock()

        snake1 = [(2, i) for i in range(2, 2 + SNAKE_LENGTH)][::-1]
        snake2 = [(7, i) for i in range(5, 5 + SNAKE_LENGTH)][::-1]

        game_over = False
        step = 0

    # --- MAIN LOOP ---
    while True:

        # --- BROADCAST state ---
        if rank == 0:
            state = {'s1': snake1, 's2': snake2, 'game_over': game_over}
        else:
            state = None

        state = comm.bcast(state, root=0)

        if state['game_over']:
            break

        my_move = None
        is_trapped = 0

        # --- WORKERS ---
        if my_role in ['s1', 's2']:
            my_body = state['s1'] if my_role == 's1' else state['s2']
            other_body = state['s2'] if my_role == 's1' else state['s1']
            head = my_body[0]

            valid_moves = get_valid_moves(head, my_body, other_body)

            if valid_moves:
                my_move = random.choice(valid_moves)
            else:
                is_trapped = 1
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dx, dy = random.choice(directions)
                my_move = (head[0] + dx, head[1] + dy)

        # --- GATHER moves ---
        all_moves = comm.gather(my_move, root=0)

        # --- REDUCE trapped ---
        total_trapped = comm.reduce(is_trapped, op=MPI.SUM, root=0)

        # --- MASTER LOGIC ---
        if rank == 0:
            step += 1

            # 🎮 HANDLE EVENTS
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            move1 = all_moves[1]
            move2 = all_moves[2]

            collision = False
            reason = ""

            if total_trapped > 0:
                collision, reason = True, "A snake was trapped!"
            elif not (0 <= move1[0] < BOARD_SIZE and 0 <= move1[1] < BOARD_SIZE):
                collision, reason = True, "Snake 1 hit wall!"
            elif not (0 <= move2[0] < BOARD_SIZE and 0 <= move2[1] < BOARD_SIZE):
                collision, reason = True, "Snake 2 hit wall!"
            elif move1 == move2:
                collision, reason = True, "Head-on collision!"
            elif move1 in state['s2']:
                collision, reason = True, "Snake 1 hit Snake 2!"
            elif move2 in state['s1']:
                collision, reason = True, "Snake 2 hit Snake 1!"
            elif move1 in state['s1']:
                collision, reason = True, "Snake 1 hit itself!"
            elif move2 in state['s2']:
                collision, reason = True, "Snake 2 hit itself!"

            if collision:
                print(f"\nGAME OVER: {reason}\n")
                game_over = True
            else:
                snake1.insert(0, move1)
                snake1.pop()
                snake2.insert(0, move2)
                snake2.pop()

            # 🎨 DRAW
            screen.fill((0, 0, 0))

            # Grid
            for x in range(BOARD_SIZE):
                for y in range(BOARD_SIZE):
                    rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(screen, (40, 40, 40), rect, 1)

            # Snake 1 (green)
            for i, (r, c) in enumerate(state['s1']):
                color = (0, 255, 0) if i == 0 else (0, 150, 0)
                pygame.draw.rect(screen, color,
                                 (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))

            # Snake 2 (red)
            for i, (r, c) in enumerate(state['s2']):
                color = (255, 0, 0) if i == 0 else (150, 0, 0)
                pygame.draw.rect(screen, color,
                                 (c*CELL_SIZE, r*CELL_SIZE, CELL_SIZE, CELL_SIZE))

            pygame.display.flip()
            clock.tick(FPS)

    # Close pygame cleanly
    if rank == 0:
        pygame.quit()

if __name__ == "__main__":
    main()