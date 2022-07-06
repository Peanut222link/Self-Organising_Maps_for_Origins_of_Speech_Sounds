from mesa import Model
from mesa.time import RandomActivation
from mesa.space import ContinuousSpace
from agent import SpeechAgent
import random

# MESA

class Model(Model):

    def __init__(self,       # ID for model
                 N,          # Initial population
                 width,      # Width of the space
                 height,     # Height of the space
                 som_size,   # Size of SOM's (22 x 22 standard)
                 ):
        self.N = N
        self.som_size = som_size
        self.schedule = RandomActivation(self)
        self.space = ContinuousSpace(width, height, True)
        self.make_agents()
        self.running = True

    def make_agents(self):
        for i in range(self.N):
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = (x, y)
            agent = SpeechAgent(
                i,
                self,
                pos,
                self.som_size)
            self.space.place_agent(agent, pos)
            self.schedule.add(agent)
            self.running = True

    def step(self):
        self.schedule.step()
        # print("\n")