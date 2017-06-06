import numpy as np

from StochasticModels.LatticeModels.MonteCarloModels.CMonteCarloModel import MonteCarloModel


class OneFactorMonteCarlo(MonteCarloModel):
    def __init__(self, one_factor_model, time_vector):
        """
        :param one_factor_model: needs to be an instance of OneFactorModel
        :return:
        """
        MonteCarloModel.__init__(self, one_factor_model, time_vector)

    def _binary_search(self, random_number, t):
        from NumericalLibrary.COptimiser import Optimiser
        vector = self.model.total_cumulative_stochastic_kernels[t][self.model.x0, :]
        x = Optimiser.binary_search(vector, random_number)
        return x

    def generate_monte_carlo_path(self, n_scenarios, update_random_matrix=True):

        if update_random_matrix or self.random_indexes is None or n_scenarios != len(self.random_indexes[:, 1]):
            self._generate_monte_carlo_indexes(n_scenarios)

        stochastic_path = np.zeros_like(self.random_indexes)
        dim = np.shape(stochastic_path)

        n_scenarios = dim[0]
        n_times = dim[1]

        for t in range(n_times):
            for s in range(n_scenarios):
                stochastic_path[s, t] = self.model.grid[self.random_indexes[s, t]]

        self.generated_path = stochastic_path
