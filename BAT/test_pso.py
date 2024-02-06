import numpy as np
from utils import Params
import matplotlib.pyplot as plt
from math import inf
from bat_optimization import BatAlgorithm


def quality_function(x):
    # Defining a quadratic cost function. The optimum is this case is trivial: [1.0, 2.0, 3.0].
    return -((x[0] - 1.0) ** 2.0 + (x[1] - 2.0) ** 2.0 + (x[2] - 3.0) ** 2.0)


# Defining hyperparameters for the algorithm
hyperparams = Params()
hyperparams.objective = "max"
hyperparams.name = "Bat Algorithm"
d = 3 # dimensionality of solution-space
n = 40 # size of population, related to amount of bees, bats and fireflies
range_min=[0.,0.,0.]
range_max = [3.,3.,3.] # solution-space range (in all dimensions)
hyperparams.d=d 
hyperparams.n=n 
hyperparams.range_min=range_min 
hyperparams.range_max=range_max 
a=0.5
r_min=0.7
r_max=1.0
alpha=0.9
gamma=0.9
f_min=0.0
f_max=5.0
hyperparams.a=a 
hyperparams.r_min=r_min 
hyperparams.r_max=r_max 
hyperparams.alpha=alpha 
hyperparams.gamma=gamma 
hyperparams.f_min=f_min 
hyperparams.f_max=f_max 
bat=BatAlgorithm(hyperparams)
position_history = []
quality_history = []
# Number of function evaluations will be 1000 times the number of particles,
# i.e. bat will be executed by 1000 generations
num_evaluations = 100 * n
for i in range(num_evaluations):
    position = bat.get_position_to_evaluate()
    value = quality_function(position)
    bat.notify_evaluation(value)
    position_history.append(np.array(position))
    quality_history.append(value)
# Finally, print the best position found by the algorithm and its value
print('Best position:', bat.get_best_position())
print('Best value:', bat.get_best_value())

fig_format = 'png'
plt.figure()
plt.plot(position_history)
plt.legend(['x[0]', 'x[1]', 'x[2]'])
plt.xlabel('Iteration')
plt.ylabel('Parameter Value')
plt.title('Parameters Convergence')
plt.grid()
plt.savefig('test_parameters_converge.%s' % fig_format, format=fig_format)
plt.figure()
plt.plot(quality_history)
plt.xlabel('Iteration')
plt.ylabel('Quality')
plt.title('Quality Convergence')
plt.grid()
plt.savefig('test_quality_converge.%s' % fig_format, format=fig_format)
best_history = []
best = -inf
for q in quality_history:
    if q > best:
        best = q
    best_history.append(best)
plt.figure()
plt.plot(best_history)
plt.xlabel('Iteration')
plt.ylabel('Best Quality')
plt.title('Best Quality Convergence')
plt.grid()
plt.savefig('test_best_convergence.%s' % fig_format, format=fig_format)
plt.show()
