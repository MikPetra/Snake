from mpi4py import MPI
import random
import time
import sys
import os

# --- Configuration ---
BOARD_SIZE = 10
SNAKE_LENGTH = 5

def get_valid_moves(head, my_body, other_body):
    """Calculates safe moves to avoid immediate crashes."""
    moves = []
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Right, Left, Down, Up
    
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

    # We need exactly 3 processes for this specific data structure
    if size != 3:
        if rank == 0:
            print("Error: Please run with exactly 3 processes.")
            print("Command: mpiexec -n 3 python mpi_snakes_collective.py")
        sys.exit()

    # ==========================================
    # 1. SCATTER: Distribute Roles at Startup
    # ==========================================
    if rank == 0:
        # Rank 0 gets 'master', Rank 1 gets 's1', Rank 2 gets 's2'
        roles_to_distribute = ['master', 's1', 's2'] 
    else:
        roles_to_distribute = None
        
    my_role = comm.scatter(roles_to_distribute, root=0)

    # Master Board Setup
    if rank == 0:
        snake1 = [(2, i) for i in range(2, 2 + SNAKE_LENGTH)][::-1] 
        snake2 = [(7, i) for i in range(5, 5 + SNAKE_LENGTH)][::-1]
        game_over = False
        step = 0

    # --- MAIN GAME LOOP ---
    while True:
        
        # Prepare state for broadcast
        if rank == 0:
            state = {'s1': snake1, 's2': snake2, 'game_over': game_over}
        else:
            state = None
            
        # ==========================================
        # 2. BROADCAST: Send board state to all Ranks
        # ==========================================
        state = comm.bcast(state, root=0)

        # If master said game over in the broadcast, break the loop
        if state['game_over']:
            break

        my_move = None
        is_trapped = 0 # 0 means safe, 1 means trapped

        # Worker Logic (Ranks 1 and 2)
        if my_role in ['s1', 's2']:
            my_body = state['s1'] if my_role == 's1' else state['s2']
            other_body = state['s2'] if my_role == 's1' else state['s1']
            head = my_body[0]

            valid_moves = get_valid_moves(head, my_body, other_body)

            if valid_moves:
                my_move = random.choice(valid_moves)
            else:
                # Trapped! Pick a fatal move and report status
                is_trapped = 1 
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                dx, dy = random.choice(directions)
                my_move = (head[0] + dx, head[1] + dy)

        # ==========================================
        # 3. GATHER: Master collects moves from workers
        # ==========================================
        # all_moves on Rank 0 will look like: [None, (x1, y1), (x2, y2)]
        all_moves = comm.gather(my_move, root=0)

        # ==========================================
        # 4. REDUCE: Sum up the 'is_trapped' status 
        # ==========================================
        # If the total sum > 0, at least one snake was trapped
        total_trapped = comm.reduce(is_trapped, op=MPI.SUM, root=0)

        # Master Game Logic (Rank 0)
        if rank == 0:
            step += 1
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"--- Step {step} ---")
            
            # Print the board
            board = [['.' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
            for r, c in state['s1']: board[r][c] = 'o'
            for r, c in state['s2']: board[r][c] = 'x'
            board[state['s1'][0][0]][state['s1'][0][1]] = 'O'
            board[state['s2'][0][0]][state['s2'][0][1]] = 'X'

            for row in board:
                print(" ".join(row))
            print("-" * 19)
            time.sleep(0.4)

            move1 = all_moves[1]
            move2 = all_moves[2]

            # Check for Collisions
            collision = False
            reason = ""

            if total_trapped > 0:
                collision, reason = True, "A snake was completely trapped and had no safe moves!"
            elif not (0 <= move1[0] < BOARD_SIZE and 0 <= move1[1] < BOARD_SIZE):
                collision, reason = True, "Snake 1 hit a wall!"
            elif not (0 <= move2[0] < BOARD_SIZE and 0 <= move2[1] < BOARD_SIZE):
                collision, reason = True, "Snake 2 hit a wall!"
            elif move1 == move2:
                collision, reason = True, "Head-on collision between Snake 1 and Snake 2!"
            elif move1 in state['s2']:
                collision, reason = True, "Snake 1 crashed into Snake 2!"
            elif move2 in state['s1']:
                collision, reason = True, "Snake 2 crashed into Snake 1!"
            elif move1 in state['s1']:
                collision, reason = True, "Snake 1 tangled into itself!"
            elif move2 in state['s2']:
                collision, reason = True, "Snake 2 tangled into itself!"

            # Update Game State
            if collision:
                print(f"\nCRASH! {reason}\n")
                game_over = True # Will be broadcasted on the next iteration
            else:
                snake1.insert(0, move1)
                snake1.pop()
                snake2.insert(0, move2)
                snake2.pop()

if __name__ == "__main__":
    main()