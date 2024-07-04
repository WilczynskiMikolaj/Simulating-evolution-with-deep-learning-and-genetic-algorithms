import csv
import random
from collections import defaultdict
import operator
import pygame
import sys

import torch

from neural_network import OrganismBrain
from settings import Settings
from organism import Organism
from food import Food
from math import floor
from numpy import std


class Simulation:
    """
    A class to represent the simulation of life.

    Attributes
    ----------
    settings : Settings
        The settings for the simulation.
    screen : pygame.Surface
        The screen where the simulation is drawn.
    clock : pygame.time.Clock
        The clock to manage simulation time.
    dt : float
        The time step for each simulation update.
    organisms : list
        The list of organisms in the simulation.
    food_list : list
        The list of food items in the simulation.
    food_left : int
        The amount of food left in the simulation.
    stats : list
        The list to store statistics of the simulation.
    gen : int
        The current generation of organisms.
    in_simulation : bool
        Flag to indicate if the simulation is running.
    start_time : int
        The start time of the simulation.
    text : str
        The text to display during simulation.
    font : pygame.font.Font
        The font used for text in the simulation.
    data_index : int
        The index of the current data set.
    entire_data : list
        The entire data set for the simulation.
    gen_max : int
        The maximum number of generations.
    food_amount : int
        The amount of food in the simulation.
    organism_amount : int
        The amount of organisms in the simulation.
    time : int
        The time duration for each generation.
    frames : int
        The number of frames for each simulation step.
    elitism_amount : float
        The proportion of the population retained as elite.
    mutations : float
        The mutation rate for organisms.
    org_speed : float
        The base speed of organisms.
    acc_factor : float
        The acceleration factor for organisms.
    rotation_factor : float
        The rotation factor for organisms.
    max_v : float
        The maximum velocity of organisms.
    csv_index : int
        The index for the CSV file.
    csv_name : str
        The name of the CSV file.
    do_not_make_more_flag : bool
        Flag to prevent creating more organisms.

    Methods
    -------
    evolve(generation, population_size):
        Evolves the population for the next generation.
    unpack_data(data):
        Unpacks the data for the simulation.
    initialize_csv():
        Initializes the CSV file for storing simulation statistics.
    save_data_to_csv(generation, food_left, best_organism, worst_organism, avg_fitness, standardDeviation, avg_v):
        Saves the simulation data to a CSV file.
    start():
        Starts the simulation loop.
    generate_organisms(number):
        Generates the initial population of organisms.
    generate_food(number):
        Generates the food items for the simulation.
    check_events():
        Checks for user input events.
    update():
        Updates the state of the simulation.
    render():
        Renders the simulation to the screen.
    render_simulation_progress(current_step, total_steps):
        Renders the progress of the simulation.
    """

    def __init__(self, data):
        """
        Constructs all the necessary attributes for the simulation.

        Parameters
        ----------
        data : list
            The data set for the simulation.
        """
        pygame.init()
        self.settings = Settings()
        self.screen = pygame.display.set_mode((self.settings.screen_width, self.settings.screen_height))
        self.clock = pygame.time.Clock()
        self.dt = 0.16
        pygame.display.set_caption("Simulation Of Life")
        self.organisms = []
        self.food_list = []
        self.food_left = 0
        self.stats = []
        self.gen = 0
        self.in_simulation: bool = True

        self.start_time = pygame.time.get_ticks()

        self.text = "Simulating..."
        self.font = pygame.font.Font(None, 74)

        self.data_index = 0

        self.entire_data = data

        # Data
        self.gen_max = 0
        self.food_amount = 0
        self.organism_amount = 0
        self.time: int = 0
        self.frames = 0
        self.elitism_amount = 0
        self.mutations = 0
        self.org_speed = 0
        self.acc_factor = 0
        self.rotation_factor = 0
        self.max_v = 0
        self.unpack_data(self.entire_data)

        self.csv_index = 0
        self.csv_name = ""

        self.do_not_make_more_flag = False
        self.initialize_csv()

    def evolve(self, generation, population_size):
        """
        Evolves the population for the next generation.

        Parameters
        ----------
        generation : int
            The current generation number.
        population_size : int
            The size of the population.
        """
        if self.in_simulation:
            print("In simulation")
        elitism_num = int(floor(self.elitism_amount * population_size))
        new_organisms = population_size - elitism_num

        fitness_stats = []
        speed_sum = 0
        stats = defaultdict(int)
        for org in self.organisms:
            if org.fitness > stats['Best'] or stats['Best'] == 0:
                stats['Best'] = org.fitness

            if org.fitness < stats['Worst'] or stats['Worst'] == 0:
                stats['Worst'] = org.fitness

            speed_sum += org._velocity
            fitness_stats.append(org.fitness)
            stats['Sum'] += org.fitness
            stats['Count'] += 1

        stats['Dev'] = std(fitness_stats)
        stats['Avg'] = stats['Sum'] / stats['Count']
        stats['AvgV'] = speed_sum / stats['Count']

        orgs_sorted = sorted(self.organisms, key=operator.attrgetter('fitness'), reverse=True)
        organisms_new = []
        for i in range(0, elitism_num):
            random_pos = (random.randint(0, 856), random.randint(0, 788))
            print(random_pos)
            organisms_new.append(Organism(self.screen, random_pos, orgs_sorted[i].name, brain=orgs_sorted[i].brain,
                                          rotation_factor=self.rotation_factor, acc_factor=self.acc_factor,
                                          basic_velocity=self.org_speed, max_v=self.max_v))

        for w in range(0, new_organisms):
            random_pos = (random.randint(0, 856), random.randint(0, 788))
            candidates = range(0, elitism_num)
            random_index = random.sample(candidates, 2)
            org_1 = orgs_sorted[random_index[0]]
            org_2 = orgs_sorted[random_index[1]]

            new_organism = Organism(self.screen, random_pos, 'gen[' + str(generation) + ']-org[' + str(w) + ']',
                                    rotation_factor=self.rotation_factor, acc_factor=self.acc_factor,
                                    basic_velocity=self.org_speed, max_v=self.max_v)
            new_organism.brain.crossover(org_1.brain, org_2.brain)
            new_organism.brain.mutate(self.mutations)
            organisms_new.append(new_organism)

        self.organisms = organisms_new
        self.stats = stats
        print("Best Organism: ", stats["Best"], "\n")
        print("Worst organism: ", stats["Worst"], "\n")

    def unpack_data(self, data):
        """
        Unpacks the data for the simulation.

        Parameters
        ----------
        data : list
            The data set for the simulation.
        """
        if self.data_index == len(self.entire_data):
            sys.exit()
        self.gen_max = data[self.data_index]["gen"]
        self.food_amount = data[self.data_index]["food"]
        self.organism_amount = data[self.data_index]["org"]
        self.time = data[self.data_index]["time"]
        self.frames = data[self.data_index]["frames"]
        self.elitism_amount = data[self.data_index]["elitism"]
        self.mutations = data[self.data_index]["mutation"]
        self.org_speed = data[self.data_index]["v"]
        self.acc_factor = data[self.data_index]["acc"]
        self.rotation_factor = data[self.data_index]["rf"]
        self.max_v = data[self.data_index]["max_v"]
        self.data_index += 1

    def initialize_csv(self):
        """
        Initializes the CSV file for storing simulation statistics.
        """
        csv_name = f"SimulationStats/simulation_stats{str(self.csv_index)}.csv"
        self.csv_index += 1
        self.csv_name = csv_name
        with open(self.csv_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Generation", "Food Left", "Best Organism", "Worst Organism", "Average fitness",
                             "Standard deviation Fitness", "Avg Speed", "Maximal Speed", "Minimal Speed"])

    def save_data_to_csv(self, generation, food_left, best_organism, worst_organism, avg_fitness,
                         standardDeviation, avg_v):
        """
        Saves the simulation data to a CSV file.

        Parameters
        ----------
        generation : int
            The current generation number.
        food_left : int
            The amount of food left.
        best_organism : int
            The fitness of the best organism.
        worst_organism : int
            The fitness of the worst organism.
        avg_fitness : float
            The average fitness of the population.
        standardDeviation : float
            The standard deviation of fitness.
        avg_v : float
            The average speed of organisms.
        """
        max_speed = 0
        min_speed = 0
        for org in self.organisms:
            min_speed = org._velocity
            if org._velocity > max_speed:
                max_speed = org._velocity
            if org._velocity < min_speed:
                min_speed = org._velocity
        with open(self.csv_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([generation, food_left, best_organism, worst_organism, avg_fitness, standardDeviation,
                             avg_v, max_speed, min_speed])

    def simulation_without_learning_start(self, org_name):
        """
        Starts the simulation loop with already trained brain.

        Parameters
        ----------
        org_name :
            Name of the brain data.
        """
        food_spawn_time = 30000
        last_food_time = pygame.time.get_ticks()
        self.generate_food(10)
        brain = OrganismBrain()
        brain.load_state_dict(torch.load(f'organism_model/{org_name}'))
        self.generate_org_with_brain(brain)
        while True:
            current_time = pygame.time.get_ticks()
            if (current_time - last_food_time) >= food_spawn_time:
                self.generate_food(10, delete=False)
                last_food_time = pygame.time.get_ticks()
            self.check_events()
            self.update()
            self.render()

    def start(self):
        """
        Starts the simulation loop.
        """
        while True:
            if self.gen < self.gen_max:
                if self.gen == 0:
                    if not self.do_not_make_more_flag:
                        self.generate_organisms(self.organism_amount)
                        self.generate_food(self.food_amount)
                        self.in_simulation = True
                        self.do_not_make_more_flag = True
                    self.check_events()
                    self.update()
                    self.render()
                    current_time = pygame.time.get_ticks()
                    if (current_time - self.start_time) >= self.time:
                        self.food_left = len(self.food_list)
                        self.generate_food(self.food_amount)
                        print("evolved")
                        self.evolve(self.gen, self.organism_amount)
                        self.save_data_to_csv(self.gen, self.food_left, self.stats["Best"], self.stats["Worst"],
                                              self.stats["Avg"], self.stats["Dev"], self.stats["AvgV"])
                        print("Current Generation: ", self.gen, "\n")
                        self.gen += 1
                        self.start_time = current_time
                elif self.gen % 10 == 0:
                    self.in_simulation = True
                    self.check_events()
                    self.update()
                    self.render()
                    current_time = pygame.time.get_ticks()
                    if (current_time - self.start_time) >= self.time:
                        self.food_left = len(self.food_list)
                        self.generate_food(self.food_amount)
                        self.evolve(self.gen, self.organism_amount)
                        print("evolved")
                        self.save_data_to_csv(self.gen, self.food_left, self.stats["Best"], self.stats["Worst"],
                                              self.stats["Avg"], self.stats["Dev"], self.stats["AvgV"])
                        print("Current Generation: ", self.gen, "\n")
                        self.gen += 1
                        self.start_time = current_time
                else:
                    self.in_simulation = False
                    frames_amount: int = 60 * self.frames
                    for i in range(frames_amount):
                        self.check_events()
                        self.update()
                        if i % 100 == 0:
                            self.render_simulation_progress(i, frames_amount)
                    self.food_left = len(self.food_list)
                    self.generate_food(self.food_amount)
                    self.evolve(self.gen, self.organism_amount)
                    self.save_data_to_csv(self.gen, self.food_left, self.stats["Best"], self.stats["Worst"],
                                          self.stats["Avg"], self.stats["Dev"], self.stats["AvgV"])
                    print("Current Generation: ", self.gen, "\n")
                    self.gen += 1
                    self.start_time = pygame.time.get_ticks()
            else:
                self.save_model()
                self.unpack_data(self.entire_data)
                self.gen = 0
                self.initialize_csv()
                self.do_not_make_more_flag = False

    def save_model(self):
        """
        Saves neural network data.
        """
        ind = 0
        for org in self.organisms:
            torch.save(org.brain.state_dict(), f"organism_model/organism_brain{str(ind)}.pth")
            ind += 1

    def generate_org_with_brain(self, brain):
        """
        Generates the organism with defined brain.

        Parameters
        ----------
        brain :
            brain of the organism.
        """
        location_x = random.randint(0, self.screen.get_width())
        location_y = random.randint(0, self.screen.get_height())
        o = Organism(self.screen, (location_x, location_y), "Organism", brain=brain,
                     rotation_factor=self.rotation_factor, acc_factor=self.acc_factor,
                     basic_velocity=self.org_speed, max_v=self.max_v)
        self.organisms.append(o)

    def generate_organisms(self, number: int):
        """
        Generates the initial population of organisms.

        Parameters
        ----------
        number : int
            The number of organisms to generate.
        """
        for i in range(number):
            location_x = random.randint(0, self.screen.get_width())
            location_y = random.randint(0, self.screen.get_height())
            o = Organism(self.screen, (location_x, location_y), "Organism: " + f'{i}',
                         rotation_factor=self.rotation_factor, acc_factor=self.acc_factor,
                         basic_velocity=self.org_speed, max_v=self.max_v)
            print("Org: ", i, "\n")
            for param in o.brain.parameters():
                print(param.data)
            self.organisms.append(o)
            print("\n\n\n")

    def generate_food(self, number: int, delete=True):
        """
        Generates the food items for the simulation.

        Parameters
        ----------
        delete: bool

        number : int
            The number of food items to generate.
        """
        if delete:
            self.food_list = []
        for i in range(number):
            location_x = random.randint(0, self.screen.get_width())
            location_y = random.randint(0, self.screen.get_height())
            self.food_list.append(Food((location_x, location_y), self.screen))

    def check_events(self):
        """
        Checks for user input events.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            for organism in self.organisms:
                organism.check_events()

    def update(self):
        """
        Updates the state of the simulation.
        """
        for organism in self.organisms:
            organism.find_nearest_food(self.food_list)
            organism.check_collision_with_food(self.food_list)
            organism.update(self.dt)

    def render(self):
        """
        Renders the simulation to the screen.
        """
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
        """
        Renders the progress of the simulation.

        Parameters
        ----------
        current_step : int
            The current step in the simulation.
        total_steps : int
            The total number of steps in the simulation.
        """
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
