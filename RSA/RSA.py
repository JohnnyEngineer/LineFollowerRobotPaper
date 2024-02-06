import time
import numpy as np

class RSA():
    def __init__(self, obj, lb, ub, dim, n, max_iter):
        self.n=n
        self.dim=dim
        self.max_iter=max_iter
        self.obj=obj
        self.ub=ub 
        self.lb=lb


        self.best_p = np.zeros((1, dim))[0]  # best positions
        if obj=="min":
            self.best_f = np.inf  # best fitness
        else:
            self.best_f = -np.inf  # best fitness

        self.x = np.zeros((n, dim))  # Initialize the positions of solution

        for i in range(dim):
            self.x[:, i] = (
                    np.random.uniform(lb[i], ub[i], n) * (ub[i] - lb[i]) + lb[i]
            )
        self.x_new = np.zeros((n, dim))

        self.t = 0  # starting iteration
        self.alpha = 0.1  # the best value 0.1
        self.beta = 0.005  # the best value 0.005
        self.f_fun = np.zeros((1, n))  # old fitness values
        self.f_fun_new = np.zeros((1, n))  # new fitness values
        if obj=="min":
            self.f_fun[0,:] = np.inf
        if obj=="max":
            self.f_fun[0,:] = -np.inf
        self.i=0
        self.posicao=0

    def get_position_to_evaluate(self):
        i=self.i
        es = 2 * np.random.randn() * (1 - (self.t / self.max_iter))
        for j in range(self.dim):
            r = self.best_p[j] - self.x[np.random.choice([0, self.n - 1]), j] / (self.best_p[j] + np.spacing(1))
            p = self.alpha + (self.x[i, j] - np.mean(self.x[i])) / (self.best_p[j] * (self.ub[j] - self.lb[j]) + np.spacing(1))
            eta = self.best_p[j] * p
            if self.t < 0.25 * self.max_iter:
                self.x_new[i, j] = self.best_p[j] - eta * self.beta - r * np.random.rand()
            elif 0.5 * self.max_iter > self.t >= 0.25 * self.max_iter:
                self.x_new[i, j] = self.best_p[j] * self.x[np.random.choice([0, self.n - 1]), j] * es * np.random.rand()
            elif 0.75 * self.max_iter > self.t >= 0.5 * self.max_iter:
                self.x_new[i, j] = self.best_p[j] * p * np.random.rand()
            else:
                self.x_new[i, j] = self.best_p[j] - eta * np.spacing(1) - r * np.random.rand()

        self.x_new[i] = np.clip(self.x_new[i], a_min=self.lb, a_max=self.ub)
        self.i+=1
        if self.i==self.n:
            self.i=0
        self.posicao=i
        return self.x_new[i]

    def notify_evaluation(self, value):
        self.execute_search_step(value)

    def execute_search_step(self, value):
        i=self.posicao
        self.f_fun_new[0, i] = value#self.f_obj(self.x_new[i])
        if self.obj=="min":
            if self.f_fun_new[0, i] < self.f_fun[0, i]:
                self.x[i] = self.x_new[i]
                self.f_fun[0, i] = self.f_fun_new[0, i]
            if self.f_fun[0, i] < self.best_f:
                self.best_f = self.f_fun[0, i]
                self.best_p = self.x[i]
        else:
            if self.f_fun_new[0, i] > self.f_fun[0, i]:
                self.x[i] = self.x_new[i]
                self.f_fun[0, i] = self.f_fun_new[0, i]
            if self.f_fun[0, i] > self.best_f:
                self.best_f = self.f_fun[0, i]
                self.best_p = self.x[i]

        print(f"At iteration {self.t} the best solution fitness is: {self.best_f}, {self.best_p}")

    def get_best_position(self):

        return self.best_p

    def get_best_value(self):
        return self.best_f