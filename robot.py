import numpy as np

class Robot:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.q_comfort = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])