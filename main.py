from simulation import Simulation

# Testing data
data = [
    # Speed and high acc
    {
        "gen": 5, "food": 30, "org": 10, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 20, "acc": 1, "rf": 360, "max_v": 1000
    },
]


if __name__ == '__main__':
    simulation = Simulation(data)
    #simulation.simulation_without_learning_start()
    simulation.start()
