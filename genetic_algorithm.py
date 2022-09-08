import pandas as pd
from time import time
from random import shuffle, seed
from numpy.random import randint, rand


class Candidate:
    def __init__(self):
        self.nationality = [
            "British couple",
            "Canadian couple",
            "Chinese businessman",
            "French lady",
            "Indian man",
        ]
        self.car_type = [
            "Holden Barina",
            "Honda Civic",
            "Hyundai Accent",
            "Nissan X-Trail",
            "Toyota Camry",
        ]
        self.car_colour = ["black", "blue", "green", "red", "white"]
        self.departure = ["5:00am", "6:00am", "7:00am", "8:00am", "9:00am"]
        self.destination = [
            "Gold Coast",
            "Newcastle",
            "Port Macquarie",
            "Sydney",
            "Tamworth",
        ]
        self.shuffle()
        self.attributes = {
            "Nationality": self.nationality,
            "Car Type": self.car_type,
            "Car Colour": self.car_colour,
            "Departure": self.departure,
            "Destination": self.destination,
        }
        self.state = pd.DataFrame(
            self.attributes, index=["Spot 1", "Spot 2", "Spot 3", "Spot 4", "Spot 5"]
        ).transpose()

    def shuffle(self):
        shuffle(self.nationality)
        shuffle(self.car_type)
        shuffle(self.car_colour)
        shuffle(self.departure)
        shuffle(self.destination)


class Fitness:
    def __init__(self, state):
        self.state = state
        self.fitness = 0

    def get_fitness(self):
        if self.state.iloc[0, 4] == "Indian man":
            self.fitness += 1
        if self.state.iloc[2, 2] == "black":
            self.fitness += 1
        # Any single column
        for y in range(5):
            if (
                self.state.iloc[0, y] == "British couple"
                and self.state.iloc[1, y] == "Toyota Camry"
                and self.state.iloc[3, y] == "6:00am"
            ):
                self.fitness += 1
            if (
                self.state.iloc[1, y] == "Hyundai Accent"
                and self.state.iloc[3, y] == "9:00am"
            ):
                self.fitness += 1
            if (
                self.state.iloc[1, y] == "Nissan X-Trail"
                and self.state.iloc[4, y] == "Sydney"
            ):
                self.fitness += 1
            if (
                self.state.iloc[3, y] == "5:00am"
                and self.state.iloc[4, y] == "Newcastle"
            ):
                self.fitness += 1
            if self.state.iloc[2, y] == "red" and self.state.iloc[4, y] == "Tamworth":
                self.fitness += 1
            if self.state.iloc[2, y] == "black" and self.state.iloc[3, y] == "8:00am":
                self.fitness += 1
            if (
                self.state.iloc[3, y] == "6:00am"
                and self.state.iloc[4, y] == "Tamworth"
            ):
                self.fitness += 1
            # Exactly to the left of
            if y != 0:
                if (
                    self.state.iloc[4, y] == "Gold Coast"
                    and self.state.iloc[0, y - 1] == "French lady"
                ):
                    self.fitness += 1
                if (
                    self.state.iloc[2, y] == "green"
                    and self.state.iloc[0, y - 1] == "Chinese businessman"
                ):
                    self.fitness += 1
                if (
                    self.state.iloc[1, y] == "Honda Civic"
                    and self.state.iloc[3, y] == "7:00am"
                    and self.state.iloc[4, y - 1] == "Gold Coast"
                ):
                    self.fitness += 1
                if (
                    self.state.iloc[0, y] == "Indian man"
                    and self.state.iloc[0, y - 1] == "Chinese businessman"
                ):
                    self.fitness += 1
            # Exactly to the right of
            if y != 4:
                if (
                    self.state.iloc[1, y] == "Holden Barina"
                    and self.state.iloc[2, y] == "blue"
                    and self.state.iloc[0, y + 1] == "British couple"
                ):
                    self.fitness += 1
                if (
                    self.state.iloc[2, y] == "white"
                    and self.state.iloc[3, y + 1] == "7:00am"
                ):
                    self.fitness += 1
        return self.fitness


def selection(population, scores, pop_size):
    selected = randint(pop_size)
    for r_candidate in randint(0, pop_size, 2):
        if scores[r_candidate] > scores[selected]:
            selected = r_candidate
    return population[selected]


def crossover(parent_1, parent_2, cross_rate):
    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    if rand() < cross_rate:
        slice = randint(1, len(parent_1) - 1)
        child_1 = pd.merge(parent_1.iloc[:slice], parent_2.iloc[slice:], how="outer")
        child_2 = pd.merge(parent_2.iloc[:slice], parent_1.iloc[slice:], how="outer")
    return [child_1, child_2]


def mutation(child, mut_rate):
    for y_1 in range(len(child)):
        for x in range(len(child)):
            if rand() < mut_rate:
                y_2 = randint(0, 5)
                swap_1 = child.iloc[x, y_1]
                swap_2 = child.iloc[x, y_2]
                child.iloc[x, y_1] = swap_2
                child.iloc[x, y_2] = swap_1
    return child


def main():
    # seed(100)
    start_time = time()

    # total generations
    generations = 0
    # population size (even populations only)
    pop_size = 130
    # crossover rate
    cross_rate = 0.9
    # mutation rate
    mut_rate = 0.05

    population = list()
    for _ in range(pop_size):
        population.append(Candidate().state)

    # initial values
    best_state = 0
    best_fitness = Fitness(population[0]).get_fitness()

    print("Now running genetic algorithm...\n")

    while best_fitness < 15:
        generations += 1
        scores = list()
        for candidate in population:
            scores.append(Fitness(candidate).get_fitness())
        for i in range(pop_size):
            if scores[i] > best_fitness:
                best_state = population[i]
                best_fitness = scores[i]
        selected = list()
        for _ in range(pop_size):
            selected.append(selection(population, scores, pop_size))
        children = list()
        for i in range(0, pop_size, 2):
            parent_1 = selected[i]
            parent_2 = selected[i + 1]
            for child in crossover(parent_1, parent_2, cross_rate):
                children.append(mutation(child, mut_rate))
        population = children
        if best_fitness == 15:  # solved
            end_time = time() - start_time
            print("\nSolved in: " + "{:.2f}".format(end_time) + " seconds")
            best_state = best_state.rename(
                index={
                    0: "Nationality",
                    1: "Car Type",
                    2: "Car Colour",
                    3: "Departure",
                    4: "Destination",
                }
            )
            print("Solution: ")
            print(best_state)
        elif generations % 5 == 0:  # current generations and fitness
            print(
                "Currently at generation %d with a fitness of %d/15"
                % (generations, best_fitness)
            )
        if generations == 50:  # possibly stuck in a local extreme
            # initialise a new population of new candidates
            print("\nPossibly reached local extreme. Initialising new population...\n")
            generations = 0
            population = list()
            for _ in range(pop_size):
                population.append(Candidate().state)
            best_state = 0
            best_fitness = Fitness(population[0]).get_fitness()


if __name__ == "__main__":
    main()
