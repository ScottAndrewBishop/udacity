# Imports
# ==================================================

import numpy as np

from bresenham import bresenham
from queue     import PriorityQueue
from enum      import (
    Enum,
    auto
)



# Classes
# ==================================================

class Action(Enum):
    WEST   = ( 0, -1, 1)
    EAST   = ( 0,  1, 1)
    NORTH  = (-1,  0, 1)
    SOUTH  = ( 1,  0, 1)
    N_WEST = (-1, -1, np.sqrt(2))
    N_EAST = (-1,  1, np.sqrt(2))
    S_WEST = ( 1, -1, np.sqrt(2))
    S_EAST = ( 1,  1, np.sqrt(2))

    @property
    def cost(self):
        return self.value[2]

    @property
    def delta(self):
        return (self.value[0], self.value[1])


class States(Enum):
    MANUAL    = auto()
    ARMING    = auto()
    TAKEOFF   = auto()
    WAYPOINT  = auto()
    LANDING   = auto()
    DISARMING = auto()
    PLANNING  = auto()



# Grid Methods
# ==================================================

def create_grid(data, drone_altitude, safety_distance):
    """
    """

    # Grid Height
    north_min  = np.floor(np.min(data[:, 0] - data[:, 3]))
    north_max  = np.ceil( np.max(data[:, 0] + data[:, 3]))
    north_size = int(np.ceil(north_max - north_min))


    # Grid Width
    east_min  = np.floor(np.min(data[:, 1] - data[:, 4]))
    east_max  = np.ceil( np.max(data[:, 1] + data[:, 4]))
    east_size = int(np.ceil(east_max  - east_min))


    # Initialize Grid
    grid = np.zeros((north_size, east_size))

    for i in range(data.shape[0]):
        north, east, alt, d_north, d_east, d_alt = data[i, :]

        # Populate Grid with Collider Objects
        if alt + d_alt + safety_distance > drone_altitude:
            obstacle = [
                int(np.clip(north - d_north - safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(north + d_north + safety_distance - north_min, 0, north_size - 1)),
                int(np.clip(east  - d_east  - safety_distance - east_min,  0, east_size  - 1)),
                int(np.clip(east  + d_east  + safety_distance - east_min,  0, east_size  - 1))
            ]

            grid[obstacle[0]:obstacle[1] + 1, obstacle[2]:obstacle[3] + 1] = 1


    return grid, int(north_min), int(east_min)



# Plan Methods
# ==================================================

def a_star(grid, h, start, goal):
    """
    """
    
    # Initialize A*
    path      = []
    path_cost = 0
    queue     = PriorityQueue()
    branch    = {}
    found     = False
    visited   = set(start)


    # Check if Goal is not Feasible
    if grid[goal] == 1:
        return path, path_cost


    # Add Start to Plan
    queue.put((path_cost, start))


    # While Paths to Explore
    while not queue.empty():
        item         = queue.get()
        current_node = item[1]


        # If Position is Start
        if current_node == start: current_cost = path_cost
        else:                     current_cost = branch[current_node][0]
            

        # If Position is Goal
        if current_node == goal:        
            found = True

            break

        
        # Otherwise Build Plan
        else:
            for action in valid_actions(grid, current_node):
                da          = action.delta
                next_node   = (current_node[0] + da[0], current_node[1] + da[1])
                branch_cost = current_cost + action.cost
                queue_cost  = branch_cost + h(next_node, goal)
                
                if next_node not in visited:                
                    visited.add(next_node)

                    branch[next_node] = (branch_cost, current_node, action)

                    queue.put((queue_cost, next_node))
             
    # If Plan Exists
    if found:
        n         = goal
        path_cost = branch[n][0]

        path.append(goal)

        while branch[n][1] != start:
            path.append(branch[n][1])

            n = branch[n][1]

        path.append(branch[n][1])


    return path[::-1], path_cost


def heuristic(position, goal_position):
    """
    """
    
    # Euclidean Distance
    return np.linalg.norm(np.array(position) - np.array(goal_position))


def valid_actions(grid, current_node):
    """
    """
    
    valid_actions = list(Action)


    # Get Current Position on Grid
    n, m = grid.shape[0] - 1, grid.shape[1] - 1
    x, y = current_node


    # Check if N, E, S, W is Feasible
    if x - 1 < 0 or grid[x - 1, y    ] == 1: valid_actions.remove(Action.NORTH)
    if x + 1 > n or grid[x + 1, y    ] == 1: valid_actions.remove(Action.SOUTH)
    if y - 1 < 0 or grid[x,     y - 1] == 1: valid_actions.remove(Action.WEST)
    if y + 1 > m or grid[x,     y + 1] == 1: valid_actions.remove(Action.EAST)


    # Check if Diagonals are Feasible
    if ((x - 1 < 0) | (y - 1 < 0)) or grid[x - 1, y - 1] == 1: valid_actions.remove(Action.N_WEST)
    if ((x - 1 < 0) | (y + 1 > m)) or grid[x - 1, y + 1] == 1: valid_actions.remove(Action.N_EAST)
    if ((x + 1 > n) | (y - 1 < 0)) or grid[x + 1, y - 1] == 1: valid_actions.remove(Action.S_WEST)
    if ((x + 1 > n) | (y + 1 > m)) or grid[x + 1, y + 1] == 1: valid_actions.remove(Action.S_EAST)


    return valid_actions



# Prune Methods
# ==================================================

def prune_path(grid, path):
    """
    """
    
    # If Plan Exists
    if path is not None:
        idx = 0

        for _ in path[:-2]:

            # Get Ray Start and End
            p0 = [path[idx    ][0], path[idx    ][1], 1]
            p1 = [path[idx + 2][0], path[idx + 2][1], 1]


            # Get Cells in Ray Trace
            cells = bresenham(
                x0 = p0[0],
                y0 = p0[1],
                x1 = p1[0],
                y1 = p1[1]
            )


            # If all Cells in Ray Feasible 
            if all([grid[c] == 0 for c in cells]):
                del path[idx + 1]


            # Otherwise Feasible Ray Does not Exist
            else:
                idx += 1


    return path



# Visdom Methods
# ==================================================

def gen_plots(velocity, path, vis):
    """
    """
    
    print(
        f"[INFO] generating plots\n"
        f"       generating path plot"
    )

    vis.line(
        X    = np.array(path['east']),
        Y    = np.array(path['north']),
        win  = 'path',
        opts = dict(
            title  = f"Mission Path",
            xlabel = 'East',
            ylabel = 'North'
        )
    )

    print("       generating altitude plot")

    vis.line(
        X    = np.array(path['seconds']),
        Y    = np.array(path['down']),
        win  = 'altitude',
        opts = dict(
            title  = f"Mission Altitude",
            xlabel = 'Time Step',
            ylabel = 'Meters'
        )
    )

    for stat in list(velocity.keys())[1:]:
        print(f"       generating velocity {stat.capitalize()} plot")
        vis.line(
            X    = np.array(velocity['seconds']),
            Y    = np.array(velocity[stat]),
            win  = stat,
            opts = dict(
                title  = f"Mission Velocity ({stat.capitalize()})",
                xlabel = 'Time Step',
                ylabel = 'Meters / Second'
            )
        )