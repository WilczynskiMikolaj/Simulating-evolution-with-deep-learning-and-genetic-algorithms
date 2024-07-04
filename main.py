from Simulation import Simulation

# Testing data
data = [
    # Testing generations
    {
        "gen": 40, "food": 50, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 20, "food": 50, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 10, "acc": 0.75, "rf": 360, "max_v": 30
    },
    {
        "gen": 15, "food": 50, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 15, "acc": 0.75, "rf": 360, "max_v": 40
    },
    {
        "gen": 10, "food": 50, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 15, "acc": 0.75, "rf": 360, "max_v": 40
    },
    {
        "gen": 5, "food": 50, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 15, "acc": 0.75, "rf": 360, "max_v": 40
    },
    # Testing food amount
    {
        "gen": 10, "food": 10, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 20, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 40, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 80, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 150, "org": 20, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    # Testing organism
    {
        "gen": 10, "food": 40, "org": 10, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 40, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },

    {
        "gen": 10, "food": 40, "org": 60, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },

    {
        "gen": 10, "food": 40, "org": 80, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    # Testing org and food
    {
        "gen": 10, "food": 80, "org": 10, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 80, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 40, "org": 40, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    # testing time
    {
        "gen": 10, "food": 30, "org": 10, "time": 5000, "frames": 5, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 30, "org": 10, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 30, "org": 10, "time": 40000, "frames": 40, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    # Testing elitism
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.05, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.5, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    # Testing mutation rate
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.03,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.4,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    # Testing elitism and mutation random
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.1, "mutation": 0.4,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.5, "mutation": 0.05,
        "v": 8, "acc": 0.75, "rf": 360, "max_v": 20
    },
    # Testing speed
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 2, "acc": 0.75, "rf": 360, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 30, "acc": 0.75, "rf": 360, "max_v": 100
    },
    # Speed and high acc
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 100, "acc": 10, "rf": 360, "max_v": 1000
    },
    # Rotation
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 10, "max_v": 20
    },
    {
        "gen": 10, "food": 10, "org": 30, "time": 20000, "frames": 20, "elitism": 0.2, "mutation": 0.1,
        "v": 8, "acc": 0.75, "rf": 1440, "max_v": 20
    },
]

if __name__ == '__main__':
    simulation = Simulation(data)
    simulation.start()
