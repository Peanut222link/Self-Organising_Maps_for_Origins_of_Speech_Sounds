from minisom import MiniSom

class PerceptionMap(MiniSom):
    "nothing"

def initialize_perception_map(size):
    #input_len of 2 because the input will be in the form of (F1, F'2)
    perception_map = PerceptionMap(size, size, 2)
    return perception_map