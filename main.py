# CPSC 481 - Logistics Robot Decision Agent main.py
# File Authors: Aidan Wallis, Nicholas Reardon, Noah Scott

from search import find_candidate_paths
from reasoning import * 
from ml import *

# Three pre-made warehouse grids for testing
warehouse_grid1 = [
['S', '.', '.', 'H', '.'],
['#', '#', '.', 'H', '.'],
['.', '.', '.', '.', '.'],
['.', 'R', 'R', '#', '.'],
['.', '.', '.', '.', 'G']
]

warehouse_grid2 = [
['#', '#', '#', '#', 'S'],
['.', '.', '.', '.', '.'],
['.', 'H', '#', 'R', '#'],
['.', 'H', '.', '.', '.'],
['.', '.', '.', '#', 'G']
]

warehouse_grid3 = [
['#', '#', 'S', 'H', '.'],
['#', '#', '.', '#', 'H'],
['.', '.', '.', '.', '.'],
['.', '#', 'R', '#', '.'],
['.', '.', '.', 'G', '.']
]

# Finds position of the start in the given grid
def initialize_start(grid):
    start_state = None
    for row_num, row in enumerate(grid):
        for col_num, symbol in enumerate(row):
            if symbol == 'S':
                start_state = (row_num, col_num)
                break
        if start_state:
            break

    return start_state

# Finds position of the goal in the given grid
def initialize_goal(grid):
    goal_state = None
    for row_num, row in enumerate(grid):
        for col_num, symbol in enumerate(row):
            if symbol == 'G':
                goal_state = (row_num, col_num)
                break
        if goal_state:
            break

    return goal_state

# Main workspace for integrating all 3 modules
if __name__ == "__main__":

    grid = warehouse_grid1  # To allow us to easily swap between multiple example grids

    start = initialize_start(grid)  # Initialize start based on selected grid
    goal = initialize_goal(grid)    # Initialize goal based on selected grid

    candidate_paths = find_candidate_paths(grid, start, goal, 3)                        # Returns candidate paths as a list of lists
    pathA, pathB, pathC = candidate_paths[0], candidate_paths[1], candidate_paths[2]    # Initializes each candidate path to its own variable

    # Currently just printing each path for demonstration purposes. Feel free to change/remove
    print("Path A:", pathA)
    print("Path B:", pathB)
    print("Path C:", pathC)