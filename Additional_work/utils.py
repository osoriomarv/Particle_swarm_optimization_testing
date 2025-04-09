"""

MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition


Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.
keywords: {Evolutionary computation;Pareto optimization;Computational complexity;Sorting;Genetic algorithms;Optimization methods;Testing;Scalability;Computer science;Mathematical model;Computational complexity;decomposition;evolutionary algorithm;multiobjective optimization;Pareto optimality;Computational complexity;decomposition;evolutionary algorithm;multiobjective optimization;Pareto optimality},

"""

# moead_utils.py
import numpy as np

class MOEAD:
    def __init__(
        self,
        problem,
        population_size=100,
        T=20,
        max_generations=200,
        n_vars=30,
        F=0.5,
        CR=1.0,
        scalarizing_func='tchebycheff',
        seed=None
    ):
        self.problem = problem
        self.population_size = population_size
        self.T = T
        self.max_generations = max_generations
        self.n_vars = n_vars
        self.F = F
        self.CR = CR
        self.scalarizing_func = self._get_scalarizing_func(scalarizing_func)
        self.rng = np.random.default_rng(seed)

        self.weights = self._generate_weight_vectors()
        self.neighbors = self._calculate_neighbors()
        self.population = self.rng.random((self.population_size, self.n_vars))
        self.objectives = np.array([self.problem.evaluate(ind) for ind in self.population])
        self.ideal_point = np.min(self.objectives, axis=0)

    def _generate_weight_vectors(self):
        """
        Weight vectors for a 2-objective problem.
        """
        return np.array([
            [i / (self.population_size - 1), 1 - i / (self.population_size - 1)]
            for i in range(self.population_size)
        ])

    def _calculate_neighbors(self):
        """
        Calculate Euclidean distances between weight vectors and return the closest neighbors.
        """
        distances = np.linalg.norm(self.weights[:, None, :] - self.weights[None, :, :], axis=2)
        return np.argsort(distances, axis=1)[:, :self.T]

    def _get_scalarizing_func(self, name):
        """
        Return the scalarizing function based on the given name.
        """
        if name == 'tchebycheff':
            return lambda f, w, z: np.max(w * np.abs(f - z))
        elif name == 'weighted_sum':
            return lambda f, w, z: np.dot(w, f)
        else:
            raise ValueError("Unsupported scalarizing function.")

    def _de_operator(self, i, k, l):
        """
        Differential Evolution operator.
        Uses index i as the base and k, l as randomly selected neighbor indices.
        """
        x_i, x_k, x_l = self.population[i], self.population[k], self.population[l]
        mutant = np.clip(x_i + self.F * (x_k - x_l), 0, 1)
        mask = self.rng.random(self.n_vars) < self.CR
        trial = np.where(mask, mutant, x_i)
        return trial

    def run(self):
        """
        Run MOEA/D optimization and return the final population and objective values.
        """
        for gen in range(self.max_generations):
            for i in range(self.population_size):
                k, l = self.rng.choice(self.neighbors[i], 2, replace=False)
                trial = self._de_operator(i, k, l)
                trial_obj = self.problem.evaluate(trial)

                # Update the ideal point
                self.ideal_point = np.minimum(self.ideal_point, trial_obj)

                for j in self.neighbors[i]:
                    f_old = self.scalarizing_func(self.objectives[j], self.weights[j], self.ideal_point)
                    f_new = self.scalarizing_func(trial_obj, self.weights[j], self.ideal_point)
                    if f_new < f_old:
                        self.population[j] = trial
                        self.objectives[j] = trial_obj
        return self.population, self.objectives
