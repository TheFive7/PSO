from pso.PSO import PSO

from pso.Cost_function import sphere
from pso.Cost_function import rosenbrock
from pso.Cost_function import rastrigin
from pso.Cost_function import weierstrass
from pso.Cost_function import michalewicz
from pso.Cost_function import griewank
from pso.Cost_function import ackley
from pso.Cost_function import schwefel

dimension = 30

bounds = [(-500, 500) for i in range(dimension)]

args = dict(cost_func=sphere, dimension=len(bounds), bounds=bounds, max_evaluations=30000, display=True, save_results=True, tol=10 ** -16)

num_runs = 20

for i in range(num_runs):
     PSO(**args, save_directory='results/PSO_'+str(i))

