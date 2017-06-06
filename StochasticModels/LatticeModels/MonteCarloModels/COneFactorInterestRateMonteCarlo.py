from StochasticModels.LatticeModels.MonteCarloModels.COneFactorMonteCarlo import OneFactorMonteCarlo
import numpy as np
import matplotlib.pyplot as plt


class OneFactorInterestRateMonteCarlo(OneFactorMonteCarlo):
    def __init__(self, one_factor_interest_rate_model, time_vector):
        self.generated_rates_path = None
        OneFactorMonteCarlo.__init__(self, one_factor_interest_rate_model, time_vector)

    def _initialise_model(self):
        OneFactorMonteCarlo._initialise_model(self)

        if len(self.model.total_discounted_stochastic_kernels) == 0:
            self.model.create_total_discounted_stochastic_kernels(self.time_vector)

        elif (self.model.total_discounted_stochastic_kernels.keys() !=
                  self.model.total_stochastic_kernels.keys()):
            self.model.total_discounted_stochastic_kernels.clear()
            self.model.create_total_discounted_stochastic_kernels()

            self.model.total_drift_adjustment.clear()
            self.model.calculate_drift_adjustment(self.time_vector)

        if len(self.model.total_drift_adjustment) == 0:
            self.model.total_drift_adjustment.clear()
            self.model.calculate_drift_adjustment()

    def generate_monte_carlo_path(self, n_scenarios, update_random_matrix=True):
        if update_random_matrix or self.random_indexes is None or n_scenarios != len(self.random_indexes[:, 1]):
            self._generate_monte_carlo_indexes(n_scenarios)

        t_vector = sorted(self.model.total_cumulative_stochastic_kernels.keys())

        stochastic_path = np.zeros_like(self.random_indexes, dtype=np.float64)  # this stores discount factors
        n_times = len(t_vector)

        for t in range(n_times):
            key = t_vector[t]
            idx = self.random_indexes[:, t]
            stochastic_path[:, t] = self.model.total_discounted_stochastic_kernels[key][self.model.x0, idx]
            stochastic_path[:, t] *= self.model.total_drift_adjustment[key]
            stochastic_path[:, t] /= self.model.total_stochastic_kernels[key][self.model.x0, idx]

        yf_vec = np.asarray(t_vector, dtype=np.float64) * 1. / 365
        stochastic_rates_path = -1. * np.log(stochastic_path) / yf_vec  # this stores rates

        self.generated_path = stochastic_path
        self.generated_rates_path = stochastic_rates_path

    def plot(self, title=""):
        fig = plt.figure()

        t_vector = sorted(self.model.total_cumulative_stochastic_kernels.keys())

        ax = fig.add_subplot(121)
        ax.set_title("Simulation")

        curve = []
        for t in t_vector:
            curve.append(self.model.underlying_curve.spot_rate(t))

        ax.plot(t_vector, curve, 'bo-')

        for s in range(np.shape(self.generated_rates_path)[0]):
            ax.plot(t_vector, self.generated_rates_path[s], 'r--')

        ax = fig.add_subplot(122)
        ax.set_title("PDF")

        for t in t_vector:
            ax.plot(self.model.grid,
                    self.model.total_stochastic_kernels[t][self.model.x0, :], label=str(t))

        ax.legend()

        plt.suptitle(title)

        plt.show()
