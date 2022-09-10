import pandas as pd
from time import time
from random import shuffle, seed
from numpy.random import randint, rand


class Candidate:
    """
    A candidate is a data frame containing randomised attributes

    ...

    Attributes
    ----------
    nationality : list[str]
        one of its attributes
    car_type : list[str]
        one of its attributes
    car_colour : list[str]
        one of its attributes
    departure : list[str]
        one of its attributes
    destination: list[str]
        one of its attributes
    attrbutes: dict[str, list[str]]
        a combination of attributes to be passed to the data frame
    state: DataFrame
        a data frame with randomly shuffled attributes

    Methods
    -------
    shuffle()
        Randomly shuffle each attribute
    """

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


def get_fitness(state):
    """Get the fitness of the candidate.

    An optimal fitness of 15 occurs when the data frame is in
    the correct order and therefore the puzzle is solved.

    Parameters
    ----------
    state: DataFrame
        A data frame of attributes to be checked
    """

    fitness = 0

    if state.iloc[0, 4] == "Indian man":
        fitness += 1
    if state.iloc[2, 2] == "black":
        fitness += 1
    # Any single column
    for y in range(5):
        if (
            state.iloc[0, y] == "British couple"
            and state.iloc[1, y] == "Toyota Camry"
            and state.iloc[3, y] == "6:00am"
        ):
            fitness += 1
        if state.iloc[1, y] == "Hyundai Accent" and state.iloc[3, y] == "9:00am":
            fitness += 1
        if state.iloc[1, y] == "Nissan X-Trail" and state.iloc[4, y] == "Sydney":
            fitness += 1
        if state.iloc[3, y] == "5:00am" and state.iloc[4, y] == "Newcastle":
            fitness += 1
        if state.iloc[2, y] == "red" and state.iloc[4, y] == "Tamworth":
            fitness += 1
        if state.iloc[2, y] == "black" and state.iloc[3, y] == "8:00am":
            fitness += 1
        if state.iloc[3, y] == "6:00am" and state.iloc[4, y] == "Tamworth":
            fitness += 1
        # Exactly to the left of
        if y != 0:
            if (
                state.iloc[4, y] == "Gold Coast"
                and state.iloc[0, y - 1] == "French lady"
            ):
                fitness += 1
            if (
                state.iloc[2, y] == "green"
                and state.iloc[0, y - 1] == "Chinese businessman"
            ):
                fitness += 1
            if (
                state.iloc[1, y] == "Honda Civic"
                and state.iloc[3, y] == "7:00am"
                and state.iloc[4, y - 1] == "Gold Coast"
            ):
                fitness += 1
            if (
                state.iloc[0, y] == "Indian man"
                and state.iloc[0, y - 1] == "Chinese businessman"
            ):
                fitness += 1
        # Exactly to the right of
        if y != 4:
            if (
                state.iloc[1, y] == "Holden Barina"
                and state.iloc[2, y] == "blue"
                and state.iloc[0, y + 1] == "British couple"
            ):
                fitness += 1
            if state.iloc[2, y] == "white" and state.iloc[3, y + 1] == "7:00am":
                fitness += 1
    return fitness


def selection(population, scores, pop_size):
    """Select a candidate from the population.

    Select three random candidates from the population and
    return the randomly selected candidiate with the best fitness.

    Parameters
    ----------
    state: DataFrame
        A data frame of attributes

    Returns
    -------
    DataFrame
        The best randomly selected candidate
    """

    selected = randint(pop_size)
    for r_candidate in randint(0, pop_size, 2):
        if scores[r_candidate] > scores[selected]:
            selected = r_candidate
    return population[selected]


def crossover(parent_1, parent_2, cross_rate):
    """Crossover two parent candidates based on probability.

    A pair of parents may be split at a random interval
    and combined to form two unique children.

    Parameters
    ----------
    parent_1: DataFrame
        A data frame of the first parent
    parent_2: DataFrame
        A data frame of the second parent
    cross_rate: int
        Probability of performing the crossover

    Returns
    -------
    DataFrame
        first crossover child of two parents or a copy of the first parent
    DataFrame
        second crossover child of two parents or a copy of the second parent
    """

    child_1 = parent_1.copy()
    child_2 = parent_2.copy()
    if rand() < cross_rate:
        slice = randint(1, len(parent_1) - 1)
        # sliced per attributes, not "spots"
        child_1 = pd.merge(parent_1.iloc[:slice], parent_2.iloc[slice:], how="outer")
        child_2 = pd.merge(parent_2.iloc[:slice], parent_1.iloc[slice:], how="outer")
    return [child_1, child_2]


def mutation(child, mut_rate):
    """Mutate a child based on probability.

    Perform a mutation on a child candidate with the amount of
    mutation depending on probability. A mutation, should it occur,
    randomly swaps the order of two elements of its attribute.

    Parameters
    ----------
    child: DataFrame
        A dataframe of attributes

    Returns
    -------
    DataFrame
        the child with mutated attributes
    """

    for y_1 in range(len(child)):
        for x in range(len(child)):
            if rand() < mut_rate:
                y_2 = randint(0, 5)
                swap_1 = child.iloc[x, y_1]
                swap_2 = child.iloc[x, y_2]
                child.iloc[x, y_1] = swap_2
                child.iloc[x, y_2] = swap_1
    return child


def genetic_algorithm(population, pop_size, cross_rate, mut_rate):
    """Run the algorithm until the solution is found.

    The algorithm will continue to run until an optimal fitness
    of 15 is found. After the 50th generation, a new population will
    be generated as it is assumed a local extreme was found.
    The time to find a solution can be highly variable.

    Parameters
    ----------
    population: list[DataFrame]
        A population of pseudo-random data frames
    pop_size: int
        The length of the population
    cross_rate: float
        The crossover rate which influences the probability of crossover
    mut_rate: float
        The mutation rate which influences the probability of mutation
    """

    # initial values
    generations = 0
    best_state = 0
    best_fitness = get_fitness(population[0])

    print("Now running genetic algorithm...\n")
    start_time = time()

    while best_fitness < 15:
        generations += 1

        scores = list()
        for candidate in population:
            scores.append(get_fitness(candidate))
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
        if (
            generations == 50 and best_fitness != 15
        ):  # possibly stuck in a local extreme
            # initialise a new population of new candidates
            print("\nStuck in local extreme. Initialising new population...\n")
            generations = 0
            population = list()
            for _ in range(pop_size):
                population.append(Candidate().state)
            best_state = 0
            best_fitness = get_fitness(population[0])


def main():
    # seed(100)

    # even populations only
    pop_size = 130
    # crossover rate
    cross_rate = 0.9
    # mutation rate
    mut_rate = 0.05

    population = list()
    for _ in range(pop_size):
        population.append(Candidate().state)

    genetic_algorithm(population, pop_size, cross_rate, mut_rate)


if __name__ == "__main__":
    main()
