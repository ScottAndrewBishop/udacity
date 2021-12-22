# Imports
# ==================================================

# Python Modules
# -------------------------
import random

from math import *


# Custom Modules
# -------------------------
import utils
import config



# Robot Classes
# ==================================================

class Robot:
    """
    """

    def __init__(self):
        """
        """
        
        
        self.measurement_range = config.MEASUREMENT_RANGE
        self.measurement_noise = config.MEASUREMENT_NOISE
        self.motion_noise      = config.MOTION_NOISE
        
        self.world_size = config.WORLD_SIZE
        self.x          = config.WORLD_SIZE / 2.0
        self.y          = config.WORLD_SIZE / 2.0
        
        self.num_landmarks = config.NUM_LANDMARKS
        self.landmarks     = []
    
    
    def rand(self):
        """
        """
        
        return random.random() * 2.0 - 1.0
    
    
    def move(self, dx, dy):
        """
        """
        
        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise
        
        if x < 0.0 or x > self.world_size or \
           y < 0.0 or y > self.world_size:
            
            return False
        
        else:
            self.x = x
            self.y = y
            
            return True


    def sense(self):
        """
        """
           
        measurements = []
        
        for i, landmark in enumerate(self.landmarks):
            dx = landmark[0] - self.x + self.rand() * self.measurement_noise
            dy = landmark[1] - self.y + self.rand() * self.measurement_noise
            
            if abs(dx) <= self.measurement_range and \
               abs(dy) <= self.measurement_range:
                measurements.append([i, dx, dy])
        
        return measurements


    def make_landmarks(self):
        """
        """
        
        self.landmarks = []
        
        for i in range(config.NUM_LANDMARKS):
            self.landmarks.append([
                round(random.random() * self.world_size),
                round(random.random() * self.world_size)
            ])


    def __repr__(self):
        """
        """
        
        return f"Robot: [x={round(self.x, 5)} y={round(self.y, 5)}]"