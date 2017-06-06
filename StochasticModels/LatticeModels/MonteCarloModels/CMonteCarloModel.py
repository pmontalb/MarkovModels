import numpy as np
import matplotlib.pyplot as plt


class MonteCarloModel:
    def __init__(self, model, monte_carlo_times):
        """
        :param model: Stochastic Model
        :param monte_carlo_times: Times at which the simulation calculates the underlying variable
        :return:
        """
        self.model = model
        self.time_vector = monte_carlo_times

        self.random_matrix = None
        self.random_indexes = None
        self.generated_path = None

        self._initialise_model()

    def _initialise_model(self):
        """ Does all the necessary calculation
        :return:
        """
        initial_value = self.model.underlying_curve.spot_rate(0)
        self.model.set_initial_state_index(initial_value)

        if len(self.model.total_stochastic_kernels) == 0:
            self.model.create_total_stochastic_kernels(self.time_vector)

        if len(self.model.total_cumulative_stochastic_kernels) == 0:
            self.model.create_total_cumulative_stochastic_kernels()

        elif (self.model.total_cumulative_stochastic_kernels.keys() !=
                  self.model.total_stochastic_kernels.keys()):
            self.model.total_cumulative_stochastic_kernels.clear()
            self.model.create_total_cumulative_stochastic_kernels()

    def _generate_random_numbers(self, n_scenarios):
        """
        Returns a vector of random numbers distributed ~ U(0, 1)
        :param n_scenarios:
        :return:
        """
        np.random.seed()
        return np.random.rand(n_scenarios)

    def _generate_monte_carlo_indexes(self, n_scenarios):
        t_vector = sorted(self.model.total_cumulative_stochastic_kernels.keys())

        if len(t_vector) == 0:
            raise RuntimeError("Produce cumulative kernels first!")

        self.random_matrix = self._generate_random_numbers(n_scenarios)
        self.random_indexes = self.binary_search(self.random_matrix, t_vector)

    def binary_search(self, random_number_vector, time_vector):
        n_scenarios = len(random_number_vector)
        n_times = len(time_vector)

        x_s_t = np.zeros((n_scenarios, n_times), dtype=np.intp)

        for t in range(n_times):
            for s in range(n_scenarios):
                x_s_t[s, t] = self._binary_search(random_number_vector[s], time_vector[t])

        return x_s_t

    def _binary_search(self, random_number, t):
        raise NotImplementedError()

    def generate_monte_carlo_path(self, n_scenarios, update_random_matrix=True):
        raise NotImplementedError()

    def plot(self, title=""):
        fig = plt.figure()

        t_vector = sorted(self.model.total_cumulative_stochastic_kernels.keys())

        ax = fig.add_subplot(121)
        ax.set_title("Simulation")

        curve = []
        for t in t_vector:
            curve.append(self.model.underlying_curve.spot_rate(t))

        ax.plot(t_vector, curve, 'bo-')

        for s in range(np.shape(self.generated_path)[0]):
            ax.plot(t_vector, self.generated_path[s], 'r--')

        ax = fig.add_subplot(122)
        ax.set_title("PDF")

        for t in t_vector:
            ax.plot(self.model.grid,
                    self.model.total_stochastic_kernels[t][self.model.x0, :], label=str(t))

        ax.legend()

        plt.suptitle(title)

        plt.show()
