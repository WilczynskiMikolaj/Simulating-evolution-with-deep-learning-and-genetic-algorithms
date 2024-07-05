import math
from random import uniform

import torch
import numpy as np
from AssetsDirectories import basic_organism
import pygame
from neural_network import OrganismBrain
from math import degrees
from math import floor
from math import radians
from math import atan2


class Organism:
    """
    Represents an organism with a neural network brain that can move and interact within a simulated environment.

    Methods
    -------
    draw()
        Draws the organism on the screen.
    check_events()
        Placeholder for event handling logic.
    think()
        Uses the neural network to process inputs and produce outputs.
    update(dt)
        Updates the organism's state including rotation, speed, and position.
    update_position(dt)
        Updates the organism's position based on its velocity and rotation.
    update_rotation(dt)
        Updates the organism's rotation based on neural network output.
    update_speed(dt)
        Updates the organism's speed based on neural network output.
    calculate_orient_to_food(food_x, food_y)
        Calculates the orientation of the organism to the nearest food.
    find_nearest_food(food_list)
        Finds the nearest food item and updates the organism's orientation and distance to it.
    check_collision(food)
        Checks if the organism has collided with a food item.
    check_collision_with_food(food_list)
        Checks for and handles collisions with food items.
    calculate_distance(pos_x, pos_y, pos_x1, pos_y1)
        Calculates the distance between two points.
    """

    def __init__(self, screen, position, name, brain=None, rotation_factor=0, acc_factor=0, basic_velocity=6, max_v=10):
        """
        Initializes the Organism with a position, brain, and various movement parameters.

        Parameters
        ----------
        screen : pygame.Surface
            The surface on which the organism will be drawn.
        position : tuple
            The initial (x, y) position of the organism.
        name : str
            The name of the organism.
        brain : OrganismBrain, optional
            The neural network brain of the organism. If None, a new brain is created.
        rotation_factor : float, optional
            The factor by which rotation is scaled.
        acc_factor : float, optional
            The factor by which acceleration is scaled.
        basic_velocity : float, optional
            The initial velocity of the organism.
        max_v : float, optional
            The maximum velocity of the organism.
        """
        self.organism_image = pygame.image.load(basic_organism)
        self.organism_rect = self.organism_image.get_rect()
        self._screen = screen
        self.name = name

        self.pos_x = position[0]
        self.pos_y = position[1]
        self._rotation = uniform(-180, 180)
        self._velocity = basic_velocity
        self._acceleration = 0

        self.nearest_food_distance = 0
        self.fitness = 0
        self.orientation_to_nearest_food = 0

        self.brain = brain
        if brain is None:
            self.brain = OrganismBrain()
        self._output = []

        self._rotation_factor = rotation_factor
        self._acceleration_factor = acc_factor
        self.max_v = max_v

        self.nearest_obstacle_distance_top = 0
        self.nearest_obstacle_distance_bottom = 0
        self.nearest_obstacle_distance_left = 0
        self.nearest_obstacle_distance_right = 0
        self.orientation_to_nearest_obstacle = 0
        self.train_obstacle_avoid = False

    def draw(self):
        """
        Draws the organism on the screen.
        """
        rotated_image = pygame.transform.rotate(self.organism_image, -self._rotation)
        rect = rotated_image.get_rect(center=(self.pos_x, self.pos_y))
        self._screen.blit(rotated_image, rect)

    def check_events(self):
        """
        Placeholder for event handling logic.
        """
        pass

    def think(self):
        """
        Uses the neural network to process inputs and produce outputs.
        """
        if self.train_obstacle_avoid:
            input_data = torch.tensor([[self.orientation_to_nearest_food, self.nearest_food_distance,
                                        self.orientation_to_nearest_obstacle,
                                        self.nearest_obstacle_distance_top, self.nearest_obstacle_distance_bottom,
                                        self.nearest_obstacle_distance_left,
                                        self.nearest_obstacle_distance_right]]).float()
            self._output = self.brain.forward(input_data).detach().numpy()
        else:
            input_data = torch.tensor([[self.orientation_to_nearest_food, self.nearest_food_distance, 0, 0, 0, 0,
                                        0]]).float()
            self._output = self.brain.forward(input_data).detach().numpy()

    def update(self, dt, obstacles):
        """
        Updates the organism's state including rotation, speed, and position.

        Parameters
        ----------
        dt : float
            The time delta since the last update.
        obstacles : list
            A list of obstacles to check for collisions.
        """
        self.think()
        self.update_rotation(dt)
        self.update_speed(dt)
        self.update_position(dt, obstacles)

    def update_position(self, dt, obstacles):
        """
        Updates the organism's position based on its velocity and rotation.

        Parameters
        ----------
        dt : float
            The time delta since the last update.
        obstacles : list
            A list of obstacles to check for collisions.
        """
        movement_dx = math.cos(math.radians(self._rotation)) * self._velocity * dt
        movement_dy = math.sin(math.radians(self._rotation)) * self._velocity * dt
        # Check new position along x-axis
        new_pos_x = self.pos_x + movement_dx
        temp_rect_x = self.organism_rect.copy()
        temp_rect_x.center = (new_pos_x, self.pos_y)

        collision_x = False
        for obstacle in obstacles:
            if temp_rect_x.colliderect(obstacle.rect):
                collision_x = True
                if self.train_obstacle_avoid and obstacle.check_name(self.name):
                    self.fitness -= 1
                    if self.fitness < 0:
                        self.fitness = 0
                break

        if not collision_x:
            self.pos_x = new_pos_x

        # Check new position along y-axis
        new_pos_y = self.pos_y + movement_dy
        temp_rect_y = self.organism_rect.copy()
        temp_rect_y.center = (self.pos_x, new_pos_y)

        collision_y = False
        for obstacle in obstacles:
            if temp_rect_y.colliderect(obstacle.rect):
                collision_y = True
                if self.train_obstacle_avoid and obstacle.check_name(self.name):
                    self.fitness -= 1
                    if self.fitness < 0:
                        self.fitness = 0
                break

        if not collision_y:
            self.pos_y = new_pos_y

        # Ensure the organism stays within the screen boundaries
        if self.pos_x < 0:
            self.pos_x = 0
        elif self.pos_x > self._screen.get_width():
            self.pos_x = self._screen.get_width()

        if self.pos_y < 0:
            self.pos_y = 0
        elif self.pos_y > self._screen.get_height():
            self.pos_y = self._screen.get_height()

    def update_rotation(self, dt):
        """
        Updates the organism's rotation based on neural network output.

        Parameters
        ----------
        dt : float
            The time delta since the last update.
        """
        self._rotation += self._output[0][0] * dt * self._rotation_factor
        if self._rotation > 180:
            self._rotation = -180
        if self._rotation < -180:
            self._rotation = 180

    def update_speed(self, dt):
        """
        Updates the organism's speed based on neural network output.

        Parameters
        ----------
        dt : float
            The time delta since the last update.
        """
        self._velocity += self._output[0][1] * self._acceleration_factor * dt
        self._acceleration = self._output[0][1]
        if self._velocity < 0:
            self._velocity = 0
        if self._velocity > self.max_v:
            self._velocity = self.max_v

    def calculate_orient_to_food(self, food_x, food_y):
        """
        Calculates the orientation of the organism to the nearest food.

        Parameters
        ----------
        food_x : float
            The x position of the food.
        food_y : float
            The y position of the food.

        Returns
        -------
        float
            The orientation of the organism to the food in the range [-1, 1].
        """
        x = food_x - self.pos_x
        y = food_y - self.pos_y
        theta = degrees(atan2(y, x)) - self._rotation
        if abs(theta) > 180:
            theta += 360
        return theta / 180

    def find_nearest_obstacle(self, obstacles):
        nearest_obstacle = None
        min_distance = float('inf')
        for obstacle in obstacles:
            distance = self.calculate_distance(self.pos_x, self.pos_y, obstacle.position[0], obstacle.position[1])
            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle

        max_possible_distance = self._screen.get_width()
        if nearest_obstacle:
            self.nearest_obstacle_distance_top = self.calculate_distance(self.pos_x, self.pos_y,
                                                                         nearest_obstacle.rect.midtop[0],
                                                                         nearest_obstacle.rect.midtop[1])
            self.nearest_obstacle_distance_top = ((2 * self.nearest_obstacle_distance_top) / max_possible_distance) - 1
            self.nearest_obstacle_distance_left = self.calculate_distance(self.pos_x, self.pos_y,
                                                                          nearest_obstacle.rect.midleft[0],
                                                                          nearest_obstacle.rect.midleft[1])
            self.nearest_obstacle_distance_left = ((
                                                           2 * self.nearest_obstacle_distance_left) / max_possible_distance) - 1
            self.nearest_obstacle_distance_bottom = self.calculate_distance(self.pos_x, self.pos_y,
                                                                            nearest_obstacle.rect.midbottom[0],
                                                                            nearest_obstacle.rect.midbottom[1])
            self.nearest_obstacle_distance_bottom = ((
                                                             2 * self.nearest_obstacle_distance_bottom) / max_possible_distance) - 1
            self.nearest_obstacle_distance_right = self.calculate_distance(self.pos_x, self.pos_y,
                                                                           nearest_obstacle.rect.midright[0],
                                                                           nearest_obstacle.rect.midright[1])
            self.nearest_obstacle_distance_right = ((
                                                            2 * self.nearest_obstacle_distance_right) / max_possible_distance) - 1
            self.orientation_to_nearest_obstacle = self.calculate_orient_to_food(nearest_obstacle.position[0],
                                                                                 nearest_obstacle.position[1])
        else:
            self.nearest_obstacle_distance_right = 0
            self.nearest_obstacle_distance_left = 0
            self.nearest_obstacle_distance_top = 0
            self.nearest_obstacle_distance_bottom = 0
            self.orientation_to_nearest_obstacle = 0

    def find_nearest_food(self, food_list):
        """
        Finds the nearest food item and updates the organism's orientation and distance to it.

        Parameters
        ----------
        food_list : list
            Food objects.
        """
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
        """
        Checks if the organism has collided with a food item.

        Parameters
        ----------
        food : Food
            The food object to check collision with.

        Returns
        -------
        bool
            True if a collision is detected, False otherwise.
        """
        if self.organism_rect.colliderect(food.food_rect):
            return True

    def check_collision_with_food(self, food_list: list):
        """
        Checks for and handles collisions with food items.

        Parameters
        ----------
        food_list : list
            Food objects.
        """
        for food in food_list:
            if self.calculate_distance(self.pos_x, self.pos_y, food.pos_x, food.pos_y) < self.organism_rect.width / 2:
                food_list.remove(food)
                self.fitness += 1
                break

    def calculate_distance(self, pos_x, pos_y, pos_x1, pos_y1):
        """
        Calculates the distance between two points using Euclidean distance.

        Parameters
        ----------
        pos_x : float
            The x position of the first point.
        pos_y : float
            The y position of the first point.
        pos_x1 : float
            The x position of the second point.
        pos_y1 : float
            The y position of the second point.

        Returns
        -------
        float
            The distance between the two points.
        """
        return math.sqrt((pos_x1 - pos_x) ** 2 + (pos_y1 - pos_y) ** 2)
