import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class OrganismBrain(nn.Module):
    """
    A neural network model representing the brain of an organism.

    This class defines a simple neural network with one hidden layer and methods for
    mutation and crossover, which are used in the genetic algorithm for evolving the organisms.

    Methods
    -------
    forward(x)
        Defines the forward pass of the neural network.
    mutate(mutation_factor)
        Applies random mutations to the weights of the neural network.
    crossover(nn1, nn2)
        Combines weights from two parent neural networks to produce offspring.
    """
    def __init__(self):
        """
        Initializes the OrganismBrain with a simple neural network architecture.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, 2),
        )

    def forward(self, x):
        """
        Forward pass of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the neural network.

        Returns
        -------
        torch.Tensor
            Output tensor from the neural network.
        """
        return self.model(x)

    def mutate(self, mutation_factor):
        """
        Applies mutation to the weights of the neural network.

        Parameters
        ----------
        mutation_factor : float
            Probability of mutation for each weight.
        """
        mutate = random.random()
        if mutate <= mutation_factor:
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
        """
        Combines weights from two parent neural networks to produce offspring.

        Parameters
        ----------
        nn1 : nn.Module
            First parent neural network.
        nn2 : nn.Module
            Second parent neural network.
        """
        with torch.no_grad():
            for param1, param2, param_offspring in zip(nn1.parameters(), nn2.parameters(),
                                                       self.parameters()):
                a = random.random()
                param_offspring.copy_(param1 * a + param2 * (1 - a))
