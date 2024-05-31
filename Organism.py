import math
import torch
import numpy as np
from AssetsDirectories import basic_organism
import pygame
from NeuralNetwork import OrganismBrain
from math import degrees
from math import floor
from math import radians
from math import atan2


class Organism:
    def __init__(self, screen, position, name, brain=None):
        self.organism_image = pygame.image.load(basic_organism)
        self.organism_rect = self.organism_image.get_rect()
        self._screen = screen
        self.name = name

        self.pos_x = position[0]
        self.pos_y = position[1]
        self._rotation = 0
        self._velocity = 6
        self._acceleration = 0

        self.nearest_food_distance = 0
        self.fitness = 0
        self.orientation_to_nearest_food = 0

        self.brain = brain
        if brain is None:
            self.brain = OrganismBrain()
        self._output = []

    def draw(self):
        rotated_image = pygame.transform.rotate(self.organism_image, -self._rotation)
        rect = rotated_image.get_rect(center=(self.pos_x, self.pos_y))
        self._screen.blit(rotated_image, rect)

    def check_events(self):
        pass

    def think(self):
        input_data = torch.tensor([[self.orientation_to_nearest_food]]).float()
        self._output = self.brain.forward(input_data).detach().numpy()

    def update(self, dt):
        self.think()
        self.update_rotation(dt)
        self.update_speed(dt)
        self.update_position(dt)

    def update_position(self, dt):
        movement_dx = math.cos(math.radians(self._rotation)) * self._velocity * dt
        movement_dy = math.sin(math.radians(self._rotation)) * self._velocity * dt
        # print(" dx:", movement_dx, " dy:", movement_dy)
        new_pos_x = self.pos_x + movement_dx
        new_pos_y = self.pos_y + movement_dy

        if new_pos_x < 0:
            new_pos_x = 0
        elif new_pos_x > 856:
            new_pos_x = 856

        if new_pos_y < 0:
            new_pos_y = 0
        elif new_pos_y > 788:
            new_pos_y = 788

        self.pos_x = new_pos_x
        self.pos_y = new_pos_y

    def update_rotation(self, dt):
        self._rotation += self._output[0][0] * dt * 360
        if self._rotation > 180:
            self._rotation = -180
        if self._rotation < -180:
            self._rotation = 180
        # print("Rotation:", self._rotation, "Rotation_change: ", self._output[0][0])

    def update_speed(self, dt):
        self._velocity += self._output[0][1] * 0.75 * dt
        if self._velocity < 0:
            self._velocity = 0
        if self._velocity > 12:
            self._velocity = 12

    # print("Speed: ", self._velocity, " Speed_change: ", self._output[0][1])

    def calculate_orient_to_food(self, food_x, food_y):
        x = food_x - self.pos_x
        y = food_y - self.pos_y
        theta = degrees(atan2(y, x)) - self._rotation
        if abs(theta) > 180:
            theta += 360
        return theta / 180

    def find_nearest_food(self, food_list):
        nearest_food = None
        min_distance = float('inf')

        for food in food_list:
            distance = self.calculate_distance(self.pos_x, self.pos_y, food.pos_x, food.pos_y)
            if distance < min_distance:
                min_distance = distance
                nearest_food = food

        if nearest_food:
            self.nearest_food_distance = min_distance
            self.orientation_to_nearest_food = self.calculate_orient_to_food(nearest_food.pos_x, nearest_food.pos_y)
        else:
            self.nearest_food_distance = 0
            self.orientation_to_nearest_food = 0

    def check_collision(self, food):
        if self.organism_rect.colliderect(food.food_rect):
            return True

    def check_collision_with_food(self, food_list):
        for food in food_list:
            if self.calculate_distance(self.pos_x, self.pos_y, food.pos_x, food.pos_y) < self.organism_rect.width / 2:
                food_list.remove(food)
                self.fitness += 1
                break

    def calculate_distance(self, pos_x, pos_y, pos_x1, pos_y1):
        return math.sqrt((pos_x1 - pos_x) ** 2 + (pos_y1 - pos_y) ** 2)
