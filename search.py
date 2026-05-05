# CPSC 481 - Logistics Robot Decision Agent search.py
# File Author: Aidan Wallis

from collections import deque

#Finds all candidate paths from start to goal of the given grid
#Paths are returned as a list of lists
#max_paths determines how many paths are returned (defaults to 3 paths)
def find_candidate_paths(grid, start, goal, max_paths=3):
    
    queue = deque([[start]]) #Initialize queue with start node
    results = []
    
    while queue:
        path = queue.popleft() 
        current_state = path[-1] #Set current state to last node in path
        
        if current_state == goal:
            results.append(path)
        else:   
            neighbors = generate_neighbors(grid, current_state)
            for neighbor in neighbors:
                if neighbor not in path:
                    new_path = list(path)
                    new_path.append(neighbor)
                    queue.append(new_path)

    return results[0:max_paths]

#Finds all valid neighbors of the given state
def generate_neighbors(grid, state):

    move_order = ['LEFT', 'RIGHT', 'DOWN', 'UP']
    valid_neighbors = []
    direction_mapping = {
        'UP': (-1, 0),
        'RIGHT': (0, 1),
        'DOWN': (1, 0),
        'LEFT': (0, -1)
    }

    for direction in move_order:
        col_direction, row_direction = direction_mapping[direction]
        new_row = state[0] + row_direction 
        new_col = state[1] + col_direction

        if new_row != -1 and new_row < len(grid):
            if new_col != -1 and new_col < len(grid[0]):
                new_neighbor = (new_row, new_col)
                if grid[new_neighbor[0]][new_neighbor[1]] != '#':
                    valid_neighbors.append(new_neighbor)

    return valid_neighbors

##########################################################
#### Everything below is for testing within this file ####
##########################################################

warehouse_grid = [
['S', '.', '.', 'H', '.'],
['#', '#', '.', 'H', '.'],
['.', '.', '.', '.', '.'],
['.', 'R', 'R', '#', '.'],
['.', '.', '.', '.', 'G']
]

#Intialize position of the start
start_state = None
for row_num, row in enumerate(warehouse_grid):
    for col_num, symbol in enumerate(row):
        if symbol == 'S':
            start_state = (row_num, col_num)
            break
    if start_state:
        break

#Intialize position of the goal
goal_state = None
for row_num, row in enumerate(warehouse_grid):
    for col_num, symbol in enumerate(row):
        if symbol == 'G':
            goal_state = (row_num, col_num)
            break
    if goal_state:
        break

if __name__ == "__main__":

    #Change max_paths value to generate more or fewer paths. max_paths defaults to show 3 paths. Value of -1 shows all found paths
    result = find_candidate_paths(warehouse_grid, start_state, goal_state, -1)

    print("Candidate Paths:")
    for path in result:
        print(path)