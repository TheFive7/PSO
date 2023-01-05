"""In evolutionary computation, DE finds a solution by iterative improvement
of a candidate solution with regard to a given measure of quality. It is
one of the most powerful optimization tools that operate on the basis of the
same  developmental  process  in  evolutionary  algorithms.  Nevertheless,
different  from  traditional  evolutionary algorithms, DE uses the scaled
differences of vectors to produce new candidate solutions in the population.
Hence, no separate probability distribution should be used to perturb the population
members. The DE is also characterized by the advantages of having few parameters and
ease of implementation. The application of DE on engineering and biomedical studies
has attracted a high level of interest, concerning its potential. Basically, DE algorithm
works through a particular sequence of stages. First, it creates an initial population
sampled uniformly at random within the search bounds. Thereafter, three components namely
mutation, crossover and  selection  are  adopted  to  evolve  the  initial  population.
The  mutation  and  crossover  are  used  to  create  new solutions, while selection
determines the solutions that will breed a new generation.  The algorithm remains
inside loop until stopping criteria are met. We implemented the original version of the algorithm[1-3].

References

    [1] Storn, R and Price, K, Differential Evolution - a Simple and\
               Efficient Heuristic for Global Optimization over Continuous Spaces,\
               Journal of Global Optimization, 1997, 11, 341 - 359.\

"""
import random
import random as rdm
import numpy as np

__author__ = "Lhassane Idoumghar"
__license__ = "MIT"
__email__ = "lhassane.idoumghar@uha.fr"
__status__ = "Development"

from pso.util import check_bounds
from pso.util import dump_results


class PSO:
    """
    :param cost_func:  a cost/loss function is a function that maps an event or values of one or more variables onto\
    a real number intuitively representing some "cost" associated with the event. For NAS, a cost function is used to\
    summarise, as a single figure of merit, how close a given design architecture is to achieving the set aims.\

    :param bounds: sequence of (min, max) pairs for each individual. None is used to specify no bound.\

    :param particles_number: algorithm optimizes a problem by maintaining a population of candidate solutions\
    and creating new candidate solutions by combining existing ones according to its simple formulae,\
    and then keeping whichever candidate solution has the best cost on the problem at hand. Simply, pop_size\
    denotes the number of population members.\

    :param dimension: number of decision variables of the cost function\

    :param F: mutation factor F is a constant from interval [0, 2]. Algorithm is also somewhat sensitive to\
    the choice of the step size F. A good initial guess is to choose F from interval [0.5, 1].\

    :param CR:  crossover probability constant from interval [0, 1]. CR, the crossover probability constant\
    from interval [0, 1] helps to maintain the diversity of the population and is rather uncritical. If the\
    parameters are correlated, high values of CR work better; and vice versa.\

    :param strategy: different search strategies to evolve the solutions.

    :param max_evaluations: the number of evaluations for evolving the\
    solutions (i.e., computational budgets)\

    :param tol: tolerance for termination.

    :param swarm: an initial population of individuals.

    :param cost: the initial cost array for the associated individuals in the population.

    :param display: whether display the results on the output or not.

    :param save_directory: the configurations are saved into a file named 'configs.json', while 'results.json' is used\
    to save the metrics. the results also contain initial population, final population and their associated\
    metrics. Here, the output formats are json and CSV, which let's the user to do post_processing\
    using an arbitrary tool.

    :param save_results: you can save all the valid generated configurations and their associated metrics\
    to the output.
    """

    def __init__(self, cost_func=None, bounds=None, particles_number=30, dimension=30, F=0.9,
                 CR=0.5, strategy=1, max_evaluations=100, tol=0.1, swarm=None, cost=None, display=True,
                 save_directory=None, save_results=False, w=-0.6031, c1=-0.6485, c2=2.6475):

        assert cost_func is not None, "Please pass a valid cost function for your optimization problems"
        assert len(bounds) == dimension, "The bounds and dimension parameters should have equal dimensions."

        if swarm is None:
            swarm = []
            for i in range(particles_number):
                particle = []
                particlePosition = []
                particleBestPosition = []
                particleVelocity = []
                for j in range(dimension):
                    if bounds[j][0] is None:
                        particlePosition.append(random.uniform(0, bounds[j][1]))  # Position in the bounds
                    elif bounds[j][1] is None:
                        particlePosition.append(random.uniform(bounds[j][0], 1.0))
                    else:
                        particlePosition.append(random.uniform(bounds[j][0], bounds[j][1]))
                    particleVelocity.append(random.uniform(0, bounds[j][1]))
                    particleBestPosition = particlePosition

                particle.append(particlePosition)
                particle.append(particleBestPosition)
                particle.append(particleVelocity)
                swarm.append(particle)

        # evaluate the individuals after initialization
        function_evaluations = 0

        if cost is None:
            cost = []
            for i in range(len(swarm)):
                particle = swarm[i]

                cost_p = cost_func(particle[0])
                function_evaluations = function_evaluations + 1

                # only valid solutions will be saved
                if type(cost_p) is float:
                    if save_results:
                        dump_results(save_directory, particle, cost_p,
                                     function_evaluations, True)
                else:
                    if save_results:
                        dump_results(save_directory, particle, cost_p,
                                     function_evaluations, True)

                cost.append(cost_p)

                if (min(cost) < tol) or (function_evaluations >= max_evaluations):
                    return

            cost = np.array(cost)
        print(cost)

        trial_individual = []
        particle = []
        # cycle through each generation

        while True:

            # cost : fonction ; np.argmin() trouve le minimum de cost
            index_min = np.argmin(cost)  # index of the best cost of the particle in cost array
            best_particle = swarm[index_min]  # best particle of current iteration

            global_best = best_particle[1]

            # cycle through each particle in the population
            for i in range(len(swarm)):
                particle = swarm[i]

                # CALCULATE OBJETIVE OF THE PARTICLE   ------- with position
                cost_p = cost_func(particle[0])

                # cost of position < cost of pbestX
                if cost_p < cost_func(particle[1]):
                    particle[1] = particle[0]

                # UPDATE GBEST
                if cost_p < cost_func(global_best):
                    particle[1] = particle[0]
                    global_best = particle[1]
                    #print("GB UPDATE: ", global_best, "     => VALUE: ", cost_func(global_best))

            # UPDATE INERTIA WEIGHT
            #w += 10 ** -5

            for i in range(len(swarm)):
                particle = swarm[i]
                r1 = rdm.random()
                r2 = rdm.random()

                for j in range(dimension):
                    # UPDATE VELOCITY (V)
                    #   V(t + 1)   =         V(t)                       Pb(t)                X(t)                        Gb(t)             X(t)
                    particle[2][j] = w * particle[2][j] + c1 * r1 * (particle[1][j] - particle[0][j]) + c2 * r2 * (global_best[j] - particle[0][j])

                    # UPDATE POSITION (X)
                    #  X(t + 1)    =     X(t)           V(t+1)
                    particle[0][j] = particle[0][j] + particle[2][j]

                    particle[0] = check_bounds(solution=particle[0], bounds=bounds)

                trial_individual = particle

            # only valid solutions will be saved
            if type(cost_p) is float:

                function_evaluations = function_evaluations + 1
                if save_results:
                    dump_results(save_directory, trial_individual, cost_func(global_best), function_evaluations, False)
                else:
                    if save_results:
                        dump_results(save_directory, trial_individual, cost_func(global_best), function_evaluations, True)

            if (min(cost) < tol) or (function_evaluations >= max_evaluations / 30):
                return
