import numpy as np
#from RSA import RSA
from Reptile_Search_Algorithm.python.RSA_obj import RSA
from utils import Params
import matplotlib.pyplot as plt
from math import inf

def quality_function(x):
    # Defining a quadratic cost function. The optimum is this case is trivial: [1.0, 2.0, 3.0].
    return -((x[0] - 1.0) ** 2.0 + (x[1] - 2.0) ** 2.0 + (x[2] - 3.0) ** 2.0)

# Number of function evaluations will be 1000 times the number of particles,
# i.e. rsa will be executed by 1000 generations
# Defining hyperparameters for the algorithm
hyperparams = Params()

# Defining the lower and upper bounds
n=40
hyperparams.objective="max"
hyperparams.name="Algorithm RSA"
hyperparams.range_min=[0.0, 0.0, 0.0]
hyperparams.range_max=[3.0, 3.0, 3.0]
hyperparams.n=n
hyperparams.d=3
hyperparams.alpha=0.1 #0.08
hyperparams.beta=0.005 #0.65
num_evaluations = 100* 10#n
hyperparams.max_iter=1
#rsa = RSA(hyperparams)
rsa=RSA(obj="max", lb=[0,0,0], ub=[3,3,3], dim=3, n=40, max_iter=1000)
np.random.seed(60)

position_history = []
quality_history = []

for i in range(num_evaluations):
    position = rsa.get_position_to_evaluate()
    print("melhor:",rsa.best_p)
    #position=rsa.get_position_to_evaluate(n=rsa.n, dim=rsa.d, max_iter=rsa.max_iter, best_p=rsa.best_p, ub=rsa.range_max, lb=rsa.range_min, alpha=rsa.alpha, beta=rsa.beta, x_new=rsa.x_new, x=rsa.x, t=rsa.t/1000, i=rsa.nova_posicao)
    #if i==500:
    #    position=[1,2,3]
    #    rsa.x_new[rsa.nova_posicao]=position
    value = quality_function(position)
    rsa.notify_evaluation(value)

    position_history.append(np.array(position))
    quality_history.append(value)
# Finally, print the best position found by the algorithm and its value
print('Best position:', rsa.get_best_position())
print('Best value:', rsa.get_best_value())

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