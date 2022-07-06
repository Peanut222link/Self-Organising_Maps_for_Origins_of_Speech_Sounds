from perception_map import initialize_perception_map
from motor_map import initialize_motor_map
from mesa import Agent
import random
import numpy as np
import matplotlib.pyplot as plt
import formula

class SpeechAgent(Agent):

    def __init__(
        self,
        unique_id,
        model,
        pos,
        som_size
    ):
        super().__init__(unique_id, model)
        self.pos = np.array(pos)
        self.som_size = som_size
        self.perception_map = initialize_perception_map(self.som_size)
        self.motor_map = initialize_motor_map(self.som_size, self.perception_map)

    def random_move(self):
        distance = random.randrange(-1, 1)
        if (distance > 0):
            distance = distance + random.random()
        else: distance = distance - random.random()
        return distance

    def is_closer(self, agent, current_closest, possible_closest):
        if (current_closest):
            # Could possibly be changed if we use get_distance but then we have to provide the agent
            current_distance = abs(agent.pos[0] - current_closest.pos[0]) + abs(agent.pos[1] - current_closest.pos[1])
            possible_distance = abs(agent.pos[0] - possible_closest.pos[0]) + abs(agent.pos[1] - possible_closest.pos[1])
            if (possible_distance < current_distance):
                return True
            else: return False
        else: return True

    def find_closest(self):
        neighbours = self.model.space.get_neighbors(self.pos, 100, False)
        closest = ()
        for x in range(len(neighbours)):
            if (self.is_closer(self, closest, neighbours[x])):
                closest = neighbours[x]
        return closest
    
    def pick_random_neuron(self):
        random_column = random.randrange(self.som_size)
        random_row = random.randrange(self.som_size)
        random_neuron = self.motor_map.get_weights()[random_column, random_row]
        return random_neuron

    def move(self):
        new_pos = (self.pos[0] + self.random_move(), self.pos[1] + self.random_move())
        self.model.space.move_agent(self, new_pos)

    def listen(self, F1, F2, F3, F4):
        second_effective_F = formula.second_effective_formant(F2, F3, F4)
        perceptual_representation = np.array([[F1, second_effective_F]])
        self.motor_map._perceptual_map.train(perceptual_representation, 1)
        # self.perception_map.train(perceptual_representation, 1)
        self.motor_map.train(perceptual_representation, 1)

    def visualize_perception_distance_map(self):
        plt.figure(figsize=(10,10))
        plt.pcolor(self.motor_map._perceptual_map.distance_map().T, cmap= 'viridis' )
        plt.colorbar()
        plt.title("(Perception map) agent: " + str(self.unique_id))
        plt.show()

    def visualize_motor_distance_map(self):
        plt.figure(figsize=(10,10))
        plt.pcolor(self.motor_map.distance_map().T, cmap= 'viridis' )
        plt.colorbar()
        plt.title("(Motor map) agent: " + str(self.unique_id))
        plt.show()
    
    def visualize_perception_map(self):
        plt.figure(figsize = (10,10))
        plt.xlim(0, 2100)
        plt.ylim(0, 2100)
        plt.scatter(self.motor_map._perceptual_map.get_weights()[:, 0], self.motor_map._perceptual_map.get_weights()[:, 1])
        plt.title("(Perception map) agent: " + str(self.unique_id))
        plt.show()

    def visualize_motor_map(self):
        two_dimensional_map = np.full((self.som_size, self.som_size, 2), np.array([0, 0]))

        for x in range(self.som_size):
            for y in range(self.som_size):
                neuron = self.motor_map.get_weights()[x, y]
                r = neuron[0]
                h = neuron[1]
                p = neuron[2]
                F1 = formula.F1(r, h, p)
                F2 = formula.F2(r, h, p)
                F3 = formula.F3(r, h, p)
                F4 = formula.F4(r, h, p)
                second_effective_F = formula.second_effective_formant(F2, F3, F4)
                perceptual_representation = np.array([[F1, second_effective_F]])
                two_dimensional_map[x, y] = perceptual_representation

        plt.figure(figsize = (10,10))
        plt.xlim(0, 2100)
        plt.ylim(0, 2100)
        plt.scatter(two_dimensional_map[:, 0], two_dimensional_map[:, 1])
        plt.title("(Motor map) agent: " + str(self.unique_id))
        plt.show()

    def get_weights(self):
        print("agent: " + str(self.unique_id))
        # print(str(self.perception_map.get_weights()))
        print(str(self.motor_map._perceptual_map.get_weights()))
        print(str(self.motor_map.get_weights()))

    def step(self):
        active = np.random.choice([True, False])
        self.move()
        if active:
            closest_agent = self.find_closest()
            random_neuron = self.pick_random_neuron()
            r = random_neuron[0]
            h = random_neuron[1]
            p = random_neuron[2]
            F1 = formula.F1(r, h, p)
            F2 = formula.F2(r, h, p)
            F3 = formula.F3(r, h, p)
            F4 = formula.F4(r, h, p)
            self.listen(F1, F2, F3, F4)
            self.motor_map.learn()
            closest_agent.listen(F1, F2, F3, F4)
        else:
            "don't do anything"