import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import random

class GeneticAlgortihm:
    def __init__(self, max_generation, crossover_count, mutation_prob, mutation_count_on_params,
                 mutation_step_size, pop_size, parent_size, param_size,  objective_function):
            
        self.pop_size = pop_size
        self.param_size = param_size
        
        self.children = np.random.rand(self.pop_size, self.param_size)
    
        assert(self.children.shape[0] == pop_size)

        self.parent_size = parent_size

        self.parent_pop = None
        self.parent_id_list = []
        
        self.obj = objective_function

        self.max_generation = max_generation
        self.crossover_count = crossover_count
        self.mutation_prob = mutation_prob
        self.mutation_count_on_params = mutation_count_on_params
        self.mutation_step_size = mutation_step_size
    
        self.run()

    def run(self):
        for step_id in range(self.max_generation):
            print("Step id: ", step_id)
            self.step()
            
    def step(self):
        self.parent_pop = self.selection_bests()
        self.get_new_generation()
        
    def selection_bests(self):
        fitnesses = {}
        fitness_list = []
        for i in range(self.children.shape[0]):
            fitnesses[i] = self.obj.fitness(self.children[i])  
            fitness_list.append(fitnesses[i])
        print("Step average pop fitness: ", sum(fitness_list)/self.pop_size)
        self.parent_id_list = [k for k, v in sorted(fitnesses.items(), key=lambda item: item[1])][:self.parent_size]
        return self.children[self.parent_id_list].copy()
    
    def get_new_generation(self):
        new_generation = []
        crosscovered_individuals = []
    
        # Crossover
        for i in range(self.crossover_count):
            parent1, parent2 = tuple(random.choices(list(self.parent_pop), weights=None, cum_weights=None, k=2)) 
            child1, child2 = self.crossover(parent1, parent2)
            crosscovered_individuals.append(child1)
            crosscovered_individuals.append(child2)
        
        assert(len(crosscovered_individuals) == self.crossover_count*2)
            
        # Mixing children and crosscovered individuals
        assert(self.pop_size > len(crosscovered_individuals))
        assert(self.pop_size > len(crosscovered_individuals) + self.parent_size)
        
        new_generation = crosscovered_individuals + list(self.parent_pop)

        for child_id, params in enumerate(list(self.children)):
            if(len(new_generation)) == self.pop_size:
                break
            if (child_id) not in self.parent_id_list:
                new_generation.append(params)
    
        assert(len(new_generation) == self.pop_size)
        
        # Mutation:
        for i in range(self.pop_size):
            if np.random.rand() < self.mutation_prob:
                new_generation[i] = self.mutation(new_generation[i])

        assert(np.array(new_generation).shape == self.children.shape)
        self.children = np.array(new_generation)
                
 
    def mutation(self,individual):
        #select a random parameter to mutate
        parameter_ids = random.choices(list(range(self.param_size)), 
                       weights=None, cum_weights=None, k=self.mutation_count_on_params)
        mean, sdv = 0, 1 # mean and standard deviation
        
        for parameter_id in parameter_ids:
            s = np.random.normal(mean, sdv)
            individual[parameter_id] = individual[parameter_id] + self.mutation_step_size*s
            
        return individual


    def crossover(self, parent1, parent2):
        #one-point crossover
        co_point=np.random.randint(0,self.param_size)
        child1=list(parent1[:co_point]) + list(parent2[co_point:])
        child2=list(parent2[:co_point]) + list(parent1[co_point:])
        child1 = np.array(child1).reshape((self.param_size,1)).T
        child2 = np.array(child2).reshape((self.param_size,1)).T
        return child1[0], child2[0]

class ObjectiveFunction:
    def __init__(self, param_size):

        diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
        diabetes_X = diabetes_X.reshape((442,param_size-1))
        diabetes_y = diabetes_y.reshape((442,1))
        
        self.X = diabetes_X
        self.Y = diabetes_y
        
        self.dataset_size = self.X.shape[0]

        self.param_size = param_size

    # Calculating the fitness value of each individual over the whole dataset.    
    def fitness(self,params):
        params = params.reshape((self.param_size,1))
        #print("Params shape: ", params.shape)
        
        coeff = np.transpose(params[:self.param_size-1][:]).reshape((1,self.param_size-1))
        #print("Coeff shape: ",coeff.shape)
        
        bias = params[-1][0].reshape((1,1))
        #print("Bias shape: ", bias.shape)
        
        sample_shape = self.X[0].reshape((self.param_size-1,1)).shape
        #print("Sample shape: ", sample_shape)
        
        dataset_loss = 0
        
        for i in range(self.dataset_size):
            sample = self.X[i].reshape((self.param_size-1,1))
            y = np.matmul(coeff,sample) + bias  #1x1
            sample_loss = np.absolute(y.item(0)-self.Y[i])**2
            dataset_loss += sample_loss
        
        average_loss = dataset_loss/self.dataset_size
               
        return average_loss


objective_function = ObjectiveFunction(11)
obj = GeneticAlgortihm(200, 8, 0.6, 1, 0.01, 70, 3, 11, objective_function)
#max_generation, crossover_count, mutation_prob, mutation_count_on_params,
#mutation_step_size, pop_size, parent_size, param_size,  objective_function
