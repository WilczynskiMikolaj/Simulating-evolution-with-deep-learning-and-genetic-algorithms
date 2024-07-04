import pygame
import AssetsDirectories


class Food:
    """
    A class to represent food in the simulation.

    Attributes
    ----------
    pos_x : float
        The x-coordinate of the food's position.
    pos_y : float
        The y-coordinate of the food's position.
    energy : int
        The energy value of the food.
    food_image : pygame.Surface
        The image representing the food.
    food_rect : pygame.Rect
        The rectangle defining the food's position and size.
    _screen : pygame.Surface
        The screen where the food will be drawn.

    Methods
    -------
    draw():
        Draws the food on the screen.
    """
    def __init__(self, position: tuple, screen):
        """
        Constructs all the necessary attributes for the food object.

        Parameters
        ----------
        position : tuple
            A tuple containing the x and y coordinates of the food's position.
        screen : pygame.Surface
            The screen where the food will be drawn.
        """
        self.pos_x = position[0]
        self.pos_y = position[1]
        self.energy: int = 1

        self.food_image = pygame.image.load(AssetsDirectories.normal_food)
        self.food_rect = self.food_image.get_rect(center=(self.pos_x, self.pos_y))
        self._screen = screen

    def draw(self):
        """
        Draws the food on the screen.
        """
        self._screen.blit(self.food_image, self.food_rect)
