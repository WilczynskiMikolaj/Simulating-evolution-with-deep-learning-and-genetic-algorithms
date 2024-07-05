from simulation import Simulation

# Testing data
data = [
    # Speed and high acc
    {
        "gen": 50, "food": 60, "org": 20, "time": 30000, "frames": 30, "elitism": 0.2, "mutation": 0.1,
        "v": 15, "acc": 0.75, "rf": 720, "max_v": 1000
    },
]


if __name__ == '__main__':
    simulation = Simulation(data)
    #simulation.simulation_without_learning_start("organism_brain0.pth")
    simulation.start()
