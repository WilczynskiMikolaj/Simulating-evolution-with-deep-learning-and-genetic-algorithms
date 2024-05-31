import random
from collections import defaultdict
import operator
import pygame
import sys
from Settings import Settings
from Organism import Organism
from Food import Food
from math import floor


class Simulation:
    def __init__(self):
        pygame.init()
        self.settings = Settings()
        self.screen = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        self.clock = pygame.time.Clock()
        self.dt = 0.16
        pygame.display.set_caption("Simulation Of Life")
        self.organisms = []
        self.food_list = []
        self.stats = []
        self.gen = 0
        self.in_simulation: bool = True

        self.start_time = pygame.time.get_ticks()

        self.text = "Simulating..."
        self.font = pygame.font.Font(None, 74)

    def evolve(self, generation, population_size):
        if self.in_simulation:
            print("In simulation")
        elitism_num = int(floor(self.settings.elitism * population_size))
        new_organisms = population_size - elitism_num

        stats = defaultdict(int)
        for org in self.organisms:
            if org.fitness > stats['Best'] or stats['Best'] == 0:
                stats['Best'] = org.fitness

            if org.fitness < stats['Worst'] or stats['Worst'] == 0:
                stats['Worst'] = org.fitness

            stats['Sum'] += org.fitness
            stats['Count'] += 1

        stats['Avg'] = stats['Sum'] / stats['Count']

        orgs_sorted = sorted(self.organisms, key=operator.attrgetter('fitness'), reverse=True)
        organisms_new = []
        for i in range(0, elitism_num):
            random_pos = (random.randint(0, 856), random.randint(0, 788))
            print(random_pos)
            organisms_new.append(Organism(self.screen, random_pos, orgs_sorted[i].name, orgs_sorted[i].brain))

        for w in range(0, new_organisms):
            random_pos = (random.randint(0, 856), random.randint(0, 788))
            candidates = range(0, elitism_num)
            random_index = random.sample(candidates, 2)
            org_1 = orgs_sorted[random_index[0]]
            org_2 = orgs_sorted[random_index[1]]

            new_organism = Organism(self.screen, random_pos, 'gen[' + str(generation) + ']-org[' + str(w) + ']')
            new_organism.brain.crossover(org_1.brain, org_2.brain)
            new_organism.brain.mutate()
            organisms_new.append(new_organism)

        self.organisms = organisms_new
        self.stats = stats
        print("Best Organism: ", stats["Best"], "\n")
        print("Worst organism: ", stats["Worst"], "\n")

    def start(self):
        self.generate_organisms(30)
        self.generate_food(75)
        print("Generation: 0\n")
        while True:
            if self.gen % 10 == 0:
                self.in_simulation = True
                self.check_events()
                self.update()
                self.render()
                current_time = pygame.time.get_ticks()
                if (current_time - self.start_time) >= 30000:
                    if self.gen < self.settings.generations:
                        self.generate_food(75)
                        self.evolve(self.gen, 25)
                        self.gen += 1
                        print("Generation: ", self.gen, "\n")
                    self.start_time = current_time
            else:
                self.in_simulation = False
                for i in range(1800):
                    self.check_events()
                    self.update()
                    if i % 180 == 0:
                        self.render_simulation_progress(i, 1800)
                self.generate_food(75)
                self.evolve(self.gen, 25)
                self.gen += 1
                print("Generation: ", self.gen, "\n")
                self.start_time = pygame.time.get_ticks()

    def generate_organisms(self, number: int):
        for i in range(number):
            location_x = random.randint(0, 856)
            location_y = random.randint(0, 788)
            o = Organism(self.screen, (location_x, location_y), "Organism: "+f'{i}')
            print("Org: ", i, "\n")
            for param in o.brain.parameters():
                print(param.data)
            self.organisms.append(o)
            print("\n\n\n")

    def generate_food(self, number: int):
        self.food_list = []
        for i in range(number):
            location_x = random.randint(0, 856)
            location_y = random.randint(0, 788)
            self.food_list.append(Food((location_x, location_y), self.screen))

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            for organism in self.organisms:
                organism.check_events()

    def update(self):
        for organism in self.organisms:
            organism.find_nearest_food(self.food_list)
            organism.check_collision_with_food(self.food_list)
            organism.update(self.dt)

    def render(self):
        self.screen.fill(self.settings.bg_color)
        organism: Organism
        for organism in self.organisms:
            organism.draw()
        food: Food
        for food in self.food_list:
            food.draw()
        pygame.display.update()
        self.dt = self.clock.tick(60) / 1000.0

    def render_simulation_progress(self, current_step, total_steps):
        self.screen.fill(self.settings.bg_color)

        text_surface = self.font.render(self.text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(
            center=(self.settings.screen_width // 2, self.settings.screen_height // 2 - 50))
        self.screen.blit(text_surface, text_rect)

        progress = current_step / total_steps
        progress_width = int(self.settings.screen_width * 0.8 * progress)
        progress_height = 50
        progress_x = (self.settings.screen_width - int(self.settings.screen_width * 0.8)) // 2
        progress_y = self.settings.screen_height // 2

        pygame.draw.rect(self.screen, (255, 255, 255),
                         [progress_x, progress_y, int(self.settings.screen_width * 0.8), progress_height], 2)

        pygame.draw.rect(self.screen, (0, 255, 0), [progress_x, progress_y, progress_width, progress_height])

        pygame.display.flip()

        self.clock.tick(60)
