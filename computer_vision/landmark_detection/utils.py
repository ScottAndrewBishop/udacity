# Imports
# ==================================================

# Python Modules
# -------------------------
import random
import numpy             as np
import matplotlib.pyplot as plt
import seaborn           as sns

from math import *


# Custom Modules
# -------------------------
import config

from Robot import Robot



# Robot Methods
# ==================================================

def make_data():
    """
    """

    # Initialize Robot
    # --------------------------------------------------    
    complete = False
    r        = Robot()
    
    
    # Initialize Landmarks
    # --------------------------------------------------
    r.make_landmarks()

    
    # While Mission Continues
    # --------------------------------------------------
    while not complete:
        data = []
        seen = [False for row in range(config.NUM_LANDMARKS)]
    
    
        # Initial Motion
        # --------------------------------------------------
        orientation = random.random() * 2.0 * pi
        dx          = cos(orientation) * config.DISTANCE
        dy          = sin(orientation) * config.DISTANCE
            
            
        # For Every Step
        # --------------------------------------------------
        for k in range(config.N - 1):
            
            # Sense at Position
            # --------------------------------------------------
            Z = r.sense()

            
            # Check all Observed Landmarks
            # --------------------------------------------------
            for i in range(len(Z)):
                seen[Z[i][0]] = True
    
    
            # Move
            # --------------------------------------------------
            while not r.move(dx, dy):
                orientation = random.random() * 2.0 * pi
                dx          = cos(orientation) * config.DISTANCE
                dy          = sin(orientation) * config.DISTANCE

                
            # Save Sensor Data
            # --------------------------------------------------
            data.append([Z, [dx, dy]])

            
        # Check if Mission Over
        # --------------------------------------------------
        complete = (sum(seen) == config.NUM_LANDMARKS)
        

    return data



def initialize_constraints():
    """
    """
    
    total_poses     = 2 * config.N
    total_landmarks = 2 * config.NUM_LANDMARKS
    dimension       = total_poses + total_landmarks
    
    omega_shape = (dimension, dimension)
    xi_shape    = (dimension, 1)
    
    omega       = np.zeros(omega_shape)
    omega[0, 0] = 1
    omega[1, 1] = 1
    
    xi       = np.zeros(xi_shape)
    xi[0, 0] = config.WORLD_SIZE / 2
    xi[1, 0] = config.WORLD_SIZE / 2
    
    return omega, xi



def slam(data):
    """
    """
    
    # Initialize Constraints
    # --------------------------------------------------
    omega, xi = initialize_constraints()
    
    
    # For Every Robot Step
    # --------------------------------------------------
    for i, step in enumerate(data):
        measure = step[0]
        motion  = step[1]
        
        
        # Measure
        # --------------------------------------------------
        for c, x, y in measure:
            j      = (config.N + c)
            update = [x, y]
            
            for k, e in enumerate(update):
                pi = 2 * i + k
                li = 2 * j + k
                
                omega[pi, pi] +=  config.MEASURE_EPS
                omega[li, li] +=  config.MEASURE_EPS
                omega[pi, li] += -config.MEASURE_EPS
                omega[li, pi] += -config.MEASURE_EPS

                xi[pi, 0] += -e / config.MEASUREMENT_NOISE
                xi[li, 0] +=  e / config.MEASUREMENT_NOISE
                
                
        # Motion
        # --------------------------------------------------
        for c, e in enumerate(motion):
            p0 = 2 * i + c
            p1 = 2 * i + c + 2
            
            omega[p0, p0] +=  config.MOTION_EPS
            omega[p1, p1] +=  config.MOTION_EPS
            omega[p0, p1] += -config.MOTION_EPS
            omega[p1, p0] += -config.MOTION_EPS

            xi[p0, 0] += -e / config.MOTION_NOISE
            xi[p1, 0] +=  e / config.MOTION_NOISE
            
            
    mu = np.dot(np.linalg.inv(omega), xi)
    
    
    return mu



def get_poses_landmarks(mu):
    """
    """

    # Get all Poses
    # --------------------------------------------------
    poses = []
    
    for i in range(config.N):
        poses.append((
            mu[2 * i    ].item(), 
            mu[2 * i + 1].item()
        ))

    
    # Get all Landmarks
    # --------------------------------------------------
    landmarks = []
    
    for i in range(config.NUM_LANDMARKS):
        landmarks.append((
            mu[2 * (config.N + i)    ].item(), 
            mu[2 * (config.N + i) + 1].item()
        ))

        
    return (
        poses, 
        landmarks
    )


def print_all(poses, landmarks):
    """
    """
    
    info = (
        f"--------------------------------------------------\n"
        f"[INFO] Estimated Poses\n"
    )
    
    for i in range(len(poses)):
        info += f"       [{', '.join([str(round(p, 3)) for p in poses[i]])}]\n"
       
    info += (
        f"\n"
        f"[INFO] Estimated Landmarks\n"
    )

    for i in range(len(landmarks)):
        info += f"       [{', '.join([str(round(l, 3)) for l in landmarks[i]])}]\n"
        
        
    print(info)



# Plot Methods
# ==================================================

def display_world(position, landmarks = None):
    """
    """   
    
    sns.set_style("dark")

    world_grid = np.zeros((
        int(config.WORLD_SIZE) + 1, 
        int(config.WORLD_SIZE) + 1
    ))

    ax   = plt.gca()
    cols = world_grid.shape[0]
    rows = world_grid.shape[1]

    ax.set_xticks([x for x in range(1, cols)], minor = True)
    ax.set_yticks([y for y in range(1, rows)], minor = True)
    
    plt.grid(
        which = "minor", 
        ls    = "-", 
        lw    = 1, 
        color = "white"
    )
    
    plt.grid(
        which = "major",
        ls    = "-",
        lw    = 2, 
        color = "white"
    )
    
    ax.text(
        position[0], 
        position[1], 
        "o", 
        ha       = "center", 
        va       = "center", 
        color    = "r", 
        fontsize = 30
    )
    
    if landmarks is not None:
        for pos in landmarks:
            if(pos != position):
                ax.text(
                    pos[0], 
                    pos[1], 
                    "x", 
                    ha       = "center", 
                    va       = "center", 
                    color    = "purple", 
                    fontsize = 20
                )
    
    plt.show()