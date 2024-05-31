import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class OrganismBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 2),
        )

    def forward(self, x):
        return self.model(x)

    def mutate(self):
        mutate = random.random()
        if mutate <= 0.1:
            mat_pick = random.randint(0, 1)

            if mat_pick == 0:
                layer = self.model[0]
                index_row = random.randint(0, 4)
                weight = layer.weight[index_row, :].clone()
                mutation_factor = random.uniform(0.9, 1.1)
                weight *= mutation_factor
                weight = weight.clamp(-1, 1)
                layer.weight[index_row, :].data.copy_(weight)

            if mat_pick == 1:
                layer = self.model[2]
                index_row = random.randint(0, 1)
                index_col = random.randint(0, 4)
                weight = layer.weight[index_row, index_col].clone()
                mutation_factor = random.uniform(0.9, 1.1)
                weight *= mutation_factor
                weight = weight.clamp(-1, 1)
                layer.weight[index_row, index_col].data.copy_(weight)

    def crossover(self, nn1: nn.Module, nn2: nn.Module):
        with torch.no_grad():
            for param1, param2, param_offspring in zip(nn1.parameters(), nn2.parameters(),
                                                       self.parameters()):
                a = random.random()
                param_offspring.copy_(param1 * a + param2 * (1 - a))
