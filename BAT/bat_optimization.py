import numpy as np
#mat.use('agg')

class BatAlgorithm():

    def __init__(self, hyperparams):
        self.solutions = []
        self.objective = hyperparams.objective
        print("Algorithm:", hyperparams.name)
        self.n = hyperparams.n # size of population
        self.d = hyperparams.d # dimensionality of solution space
        
        self.range_min = hyperparams.range_min # lower bound in all dimensions
        self.range_max = hyperparams.range_max # upper bound in all dimensions
                   
        self.a_init = hyperparams.a # initial loudness of all bats
        self.r_max = hyperparams.r_max # maximum pulse rate of bats
        self.alpha = hyperparams.alpha # loudness decreasing factor
        self.gamma = hyperparams.gamma # pulse rate increasing factor
        self.f_min = hyperparams.f_min # minimum sampled frequency
        self.f_max = hyperparams.f_max # maximum sampled frequency
        self.posicao_utilizada=0
        self.nova_posicao=None
        self.valores= [-np.inf] * self.n
        #self.best_solution = self.random_uniform_in_ranges()
        self.best_solution = [0.0, 10.0, 0.0, 0.0]
        self.initialize()
        self.t=0
    
    
    def initialize(self):
        self.solutions = np.zeros(shape=(self.n, self.d))
        for i in range(self.n):
            self.solutions[i] = self.random_uniform_in_ranges()
            
        self.q = np.zeros(self.n)
        self.v = np.zeros((self.n, self.d))
        self.b = np.zeros((self.n, self.d))
            
        self.a = np.repeat(self.a_init, self.n)
        self.r = np.zeros(self.n)
        
    def execute_search_step(self, valor):
        
        for i in range(self.n):
            if self.compare_objective_value(valor, self.valores[i]) < 0:
                if np.random.uniform(0, 1) < self.a[i]:
                    self.solutions[i] = self.nova_posicao
                    self.valores[i]=valor
                    self.a[i] *= self.alpha
                    self.r[i] = self.r_max * (1-np.exp(-self.gamma * self.t))
        self.t+=1
    
    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        f = np.random.uniform(self.f_min, self.f_max, 1)
        nova_posicao=None
        if np.random.uniform(0, 1) < self.r[self.posicao_utilizada]:
            nova_posicao = self.best_solution + np.mean(self.a) * np.random.uniform(-1, 1, self.d)
        else:
            self.v[self.posicao_utilizada] = self.v[self.posicao_utilizada] + (self.solutions[self.posicao_utilizada] - self.best_solution) * f
            nova_posicao = self.solutions[self.posicao_utilizada] + self.v[self.posicao_utilizada]
            
        nova_posicao = self.clip_to_ranges(nova_posicao)
        #posicao=self.solutions[self.posicao_utilizada]
        self.nova_posicao=nova_posicao
        self.posicao_utilizada+=1
        if self.posicao_utilizada==self.n-1:
            #posicao=self.solutions[self.posicao_utilizada]
            self.posicao_utilizada=0
        #return posicao
        return nova_posicao

    
    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        #comparar o valor local recebido com o armazenado e modificar caso seja melhor 
        self.execute_search_step(value)

    def get_best_position(self):

        if self.objective == 'min':
            candidate = np.argmin(self.valores)
        elif self.objective == 'max':
            candidate = np.argmax(self.valores)
        if self.best_solution is None:
            self.best_solution = np.copy(self.solutions[candidate])
        elif self.best_solution in self.solutions:
            indice=np.where(self.solutions==self.best_solution)[0][0]
            if self.compare_objective_value(self.valores[candidate], self.valores[indice]) < 0:
                self.best_solution = np.copy(self.solutions[candidate])
        else:
            self.best_solution = np.copy(self.solutions[candidate])
        return self.best_solution

    def get_best_value(self):
        if self.objective == 'min':
            return np.min(self.valores)
        elif self.objective == 'max':
            return np.max(self.valores)

    def compare_objective_value(self, v0, v1):

        if self.objective == 'min':
            return v0 - v1
        elif self.objective == 'max':
            return v1 - v0
 
    
    def random_uniform_in_ranges(self):
        rnd = np.zeros(self.d)
        for i in range(self.d):
            rnd[i] = np.random.uniform(self.range_min[i], self.range_max[i])
        return rnd
    
    def clip_to_ranges(self, x):
        for i in range(self.d):
            x[i] = np.clip(x[i], self.range_min[i], self.range_max[i])
        return x
        
