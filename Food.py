import pygame
import AssetsDirectories


class Food:
    def __init__(self, position: tuple, screen):
        self.pos_x = position[0]
        self.pos_y = position[1]
        self.energy: int = 1

        self.food_image = pygame.image.load(AssetsDirectories.normal_food)
        self.food_rect = self.food_image.get_rect(center=(self.pos_x, self.pos_y))
        self._screen = screen

    def draw(self):
        self._screen.blit(self.food_image, self.food_rect)
