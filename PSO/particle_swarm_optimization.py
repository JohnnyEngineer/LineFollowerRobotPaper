import numpy as np
import random
from math import inf


class Particle:
    """
    Represents a particle of the Particle Swarm Optimization algorithm.
    """
    def __init__(self,posicao_inicial,numero_dimensoes, upper,lower):
        """
        Creates a particle of the Particle Swarm Optimization algorithm.

        :param posicao_inicial: criação dos pontos iniciais e com melhor chute possível.
        :type list.
        :param numero_dimensoes: é o número de dimensões utilizadas aqui. Serve para qualquer equação.
        :type integer.
        :param lower_bound: lower bound of the particle position.
        :type lower_bound: numpy array.
        :param upper_bound: upper bound of the particle position.
        :type upper_bound: numpy array.
        """
        self.numero_dimensoes=numero_dimensoes
        self.posicao_i=[]           # posição da partícula 
        self.velocidade_i=[]        # velocidade da partícula
        self.melhor_posicao_i=[]    # melhor posição individual
        self.melhor_erro_i=-1       # melhor erro individual
        self.erro_i=-1              # erro individual

        for i in range(0,numero_dimensoes):
            self.velocidade_i.append(random.uniform(-1,1))
            self.posicao_i.append(posicao_inicial[i])

def bounds(lower_bound, upper_bound):
    """
    Cria uma lista de tuplas para cada dimensão, com seu mínimo e seu máximo.
    Facilita o uso posterior
    :param lower_bound: lista do limite mínimo da função por dimensão. Ex.:
    [0,0,0] para três variáveis.
    :param upper_bound:lista do limite máximo da função por dimensão. Ex.:
    [3,3,3] para três variáveis.
    :return: lista de tuplas com a posição de mínimo e máximo para cada variável.
    Ex.: supondo três variáveis, retorna [(0,3),(0,3),(0,3)]
    :rtype: lista de tuplas.
    """
    lista=[]
    for i in range(len(upper_bound)):
        lista.append((lower_bound[i], upper_bound[i]))
    return lista
class ParticleSwarmOptimization:
    """
    Represents the Particle Swarm Optimization algorithm.
    Hyperparameters:
        inertia_weight: inertia weight.
        cognitive_parameter: cognitive parameter.
        social_parameter: social parameter.

    :param hyperparams: hyperparameters used by Particle Swarm Optimization.
    :type hyperparams: Params.
    :param lower_bound: lower bound of particle position.
    :type lower_bound: numpy array.
    :param upper_bound: upper bound of particle position.
    :type upper_bound: numpy array.
    """
    def __init__(self, hyperparams, lower_bound, upper_bound):
        self.particula_utilizada=None
        self.bounds=bounds(lower_bound, upper_bound)
        self.num_particles=hyperparams.num_particles
        self.melhor_erro_global=-1                   # melhor erro global
        self.melhor_posicao_global=[]                # melhor posição global
        # criar o enxame
        self.enxame=[]
        self.w= hyperparams.inertia_weight
        self.c1=hyperparams.cognitive_parameter
        self.c2=hyperparams.social_parameter
        self.numero_dimensoes=len(lower_bound)
        for i in range(0,self.num_particles):
            self.enxame.append(Particle(upper_bound.tolist(), \
            #self.enxame.append(Particle([10]*self.num_particles, \
            len(lower_bound), \
            upper_bound,lower_bound))
    

    def get_best_position(self):
        """
        Obtains the best position so far found by the algorithm.

        :return: the best position.
        :rtype: numpy array.
        """
        return self.melhor_posicao_global

    def get_best_value(self):
        """
        Obtains the value of the best position so far found by the algorithm.

        :return: value of the best position.
        :rtype: float.
        """
        # Todo: implement
        return self.melhor_erro_global

    def get_position_to_evaluate(self):
        """
        Obtains a new position to evaluate.

        :return: position to evaluate.
        :rtype: numpy array.
        """
        if self.particula_utilizada is None:
            self.particula_utilizada=0
            return self.enxame[self.particula_utilizada].posicao_i
        else:
            self.particula_utilizada+=1
            if self.particula_utilizada<len(self.enxame):
                return self.enxame[self.particula_utilizada].posicao_i
            else:
                self.particula_utilizada=0
                return self.enxame[self.particula_utilizada].posicao_i


    def advance_generation(self):
        """
        Advances the generation of particles. Auxiliary method to be used by notify_evaluation().
        """
        # Verificar a partícula escolhida e atualizar a velocidade e posição.

        for i in range(0,self.numero_dimensoes):
            #atualizar velocidade
            r1=random.random()
            r2=random.random()

            cognicao=self.c1*r1*(self.enxame[self.particula_utilizada].melhor_posicao_i[i]-self.enxame[self.particula_utilizada].posicao_i[i])
            valor_social=self.c2*r2*(self.melhor_posicao_global[i]-self.enxame[self.particula_utilizada].posicao_i[i])
            self.enxame[self.particula_utilizada].velocidade_i[i]=self.w*self.enxame[self.particula_utilizada].velocidade_i[i]+cognicao+valor_social
            #atualizar posição
            self.enxame[self.particula_utilizada].posicao_i[i]=self.enxame[self.particula_utilizada].posicao_i[i]+self.enxame[self.particula_utilizada].velocidade_i[i]

            # Ajustar o limite máximo, se necessário. Caso esteja fora, substitua pelo limite máximo.
            if self.enxame[self.particula_utilizada].posicao_i[i]>self.bounds[i][1]:
                self.enxame[self.particula_utilizada].posicao_i[i]=self.bounds[i][1]

            #  Ajustar o limite mínimo, se necessário. Caso esteja fora, substitua pelo limite mínimo.
            if self.enxame[self.particula_utilizada].posicao_i[i] < self.bounds[i][0]:
                self.enxame[self.particula_utilizada].posicao_i[i]=self.bounds[i][0]

    def notify_evaluation(self, value):
        """
        Notifies the algorithm that a particle position evaluation was completed.

        :param value: quality of the particle position.
        :type value: float.
        """
        #comparar o valor local recebido com o armazenado e modificar caso seja melhor 
        self.enxame[self.particula_utilizada].erro_i=value
        if self.enxame[self.particula_utilizada].erro_i > self.enxame[self.particula_utilizada].melhor_erro_i or self.enxame[self.particula_utilizada].melhor_erro_i==-1:
            self.enxame[self.particula_utilizada].melhor_posicao_i=self.enxame[self.particula_utilizada].posicao_i
            self.enxame[self.particula_utilizada].melhor_erro_i=self.enxame[self.particula_utilizada].erro_i
        #comparar o valor global recebido com o armazenado e modificar caso seja melhor 
        if self.enxame[self.particula_utilizada].erro_i > self.melhor_erro_global or self.melhor_erro_global == -1:
            self.melhor_posicao_global=list(self.enxame[self.particula_utilizada].posicao_i)
            self.melhor_erro_global=float(self.enxame[self.particula_utilizada].erro_i)
        self.advance_generation()


