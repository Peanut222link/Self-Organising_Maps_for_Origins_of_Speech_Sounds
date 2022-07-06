from minisom import MiniSom, asymptotic_decay
from warnings import warn
import random
import numpy as np

class MotorMap(MiniSom):
    def __init__(self, x, y, input_len, perceptual_map, sigma=1.0, learning_rate=0.5,
                 decay_function=asymptotic_decay,
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None):
        if sigma >= x or sigma >= y:
            warn('Warning: sigma is too high for the dimension of the map.')

        self._random_generator = np.random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._x = x
        self._y = y
        self._input_len = input_len
        self._perceptual_map = perceptual_map
        # random initialization
        self._weights = self._random_generator.rand(x, y, input_len)
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = np.zeros((x, y))
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)  # used to evaluate the neighborhood function

        self.weights = self.create_weights_arrays()
        self.activation_history_G_i = self.create_activation_history()
        self.activation_history_G_j = self.create_activation_history()
        self.activation_counter = 0
        self.current_G_i = np.array([])
        self.current_G_j = np.array([])

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology
        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        if topology == 'hexagonal':
            self._xx[::-2] -= 0.5
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        self._decay_function = decay_function

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))

        if neighborhood_function in ['triangle',
                                     'bubble'] and (divmod(sigma, 1)[1] != 0
                                                    or sigma < 1):
            warn('sigma should be an integer >=1 when triangle or bubble' +
                 'are used as neighborhood function')

        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys())))

            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def train(self, data, num_iteration,
              random_order=False, verbose=False, use_epochs=False):
        self._check_iteration_number(num_iteration)
        random_generator = None
        if random_order:
            random_generator = self._random_generator
        def get_decay_rate(iteration_index, data_len):
            return int(iteration_index)
        if use_epochs:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index / data_len)
        def summation(j, s):
            i_rows = 13
            i_columns = 13
            result = 0
            for x in range(i_rows):
                for y in range(i_columns):
                    result += self.weights[x * i_rows + y, j[0] * i_columns + j[1]] * self._perceptual_map.activate(s)[x, y]
            return result

        def G_j(j, s):
            result = 1 / (1 + np.e ** -(summation(j, s)))
            return result
            
        activation_map = np.full((self._x, self._y), np.empty)
        for x in range(self._x):
            for y in range(self._y):
                activation_map[x, y] = G_j([x, y], data)

        self.current_G_i = self._perceptual_map.activate(data)
        self.current_G_j = activation_map
        winner_pos = np.unravel_index(np.argmax(activation_map), activation_map.shape)
        winner_weight = np.array([self.get_weights()[winner_pos[0], winner_pos[1]]])
        decay_rate = get_decay_rate(1, len(data))
        self.update(winner_weight, winner_pos, 1, num_iteration)

    def learn(self):
        if ((self.current_G_i.size != 0) and (self.current_G_j.size != 0)):
            self.update_activation_history()
            G_i_result = self.current_G_i - self.activation_history_G_i
            G_j_result = self.current_G_j - self.activation_history_G_j
            hebbian_result = np.multiply(np.outer(G_i_result, G_j_result), 0.00001)
            self.weights += hebbian_result
    
    def create_weights_arrays(self):
                # Each neuron in the perceptual map is connected unidirectionally to all the neurons in the motor map
                # random.uniform() will give all elements in the 4D array a small random value
                weights_array = np.full((self._x * self._y, self._x * self._y), np.empty)
                for x1y1 in range(self._x * self._y):
                    for x2y2 in range(self._x * self._y):
                        weights_array[x1y1, x2y2] = random.uniform(0.000001, 0.00001)
                return weights_array

    
    def create_activation_history(self):
        activation_history = np.full((self._x, self._y), 0.00)
        return activation_history
    
    def update_activation_history(self):
        if (self.activation_counter == 0):
            self.activation_history_G_i = self.current_G_i
            self.activation_history_G_j = self.current_G_j
        if (self.activation_counter > 0):
            for x in range(self._x):
                for y in range(self._y):
                    alpha = 0.125
                    self.activation_history_G_i[x, y] = alpha * self.current_G_i[x, y] + (1 - alpha) * self.activation_history_G_i[x, y]
                    self.activation_history_G_j[x, y] = alpha * self.current_G_j[x, y] + (1 - alpha) * self.activation_history_G_j[x, y]

def initialize_motor_map(size, perceptual_map):
    # input_len of 3 because the input will be in the form of (r, h, p)
    motor_map = MotorMap(size, size, 3, perceptual_map)
    return motor_map