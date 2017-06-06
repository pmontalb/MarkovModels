import numpy as np
import scipy.linalg as linalg

from StochasticModels.LatticeModels.CLatticeModel import LatticeModel


class OneFactorModel(LatticeModel):
    """
    In particularising the global model, one-factor models operate on a single-dimensional grid. For the time being the
    grid is assumed to be a linear space, but it can be modified accordingly, as you maintain the space gradient
    under control.

    The base generator is built by solving a 2-by-2 linear system for each grid point.
    """

    def __init__(self, d, x_min, x_max):
        """
        :param d: Number of grid points
        :param x_min: Grid left bound
        :param x_max: Grid right bound
        :return:
        """
        self.drift = np.zeros((d, 1))
        self.volatility = np.zeros((d, 1))
        self.x0 = 0

        super(OneFactorModel, self).__init__(d)
        self.build_grid(x_min, x_max)

    def build_grid(self, x_min, x_max):
        # TODO: Non-uniform grid
        self.grid = np.linspace(x_min, x_max, self.d)

    def create_markov_operator(self, base_operator, dt, method="CrankNicolson"):
        u = super(OneFactorModel, self).create_markov_operator(base_operator, dt, method)

        # TODO: implement other Boundary Conditions. Be sure the grid is very large in terms of bounds

        return u

    def build_gradient_matrix(self, x):
        """
        :param x: grid index
        :return:
        """
        if x == 0 or x == self.d - 1:
            raise ValueError("Boundary Conditions not treated here")

        nabla_matrix = np.zeros((2, 2))

        nabla_minus = self.grid[x - 1] - self.grid[x]
        nabla_plus = self.grid[x + 1] - self.grid[x]

        nabla_matrix[0, 0] = nabla_minus
        nabla_matrix[0, 1] = nabla_plus
        nabla_matrix[1, 0] = nabla_minus * nabla_minus
        nabla_matrix[1, 1] = nabla_plus * nabla_plus

        if linalg.det(nabla_matrix) == 0:
            raise ValueError("Nabla is not invertible")

        return nabla_matrix

    def build_base_operator(self, t):
        """
        :param t: Not used as mu and sigma are constant
        :return:
        """
        # Update drift and volatility
        self.build_moment_vectors(t)

        base_operator = np.zeros((self.d, self.d))

        nabla = linalg.block_diag(*[self.build_gradient_matrix(x) for x in range(1, self.d - 1)])

        moments = np.zeros(2 * (self.d - 2))
        for i in range(0, self.d - 2):
            moments[2 * i] = self.drift[i + 1]
            moments[2 * i + 1] = self.volatility[i + 1]

        generator_elements = linalg.solve(nabla, moments)

        r_idx, c_idx = np.diag_indices_from(base_operator)
        base_operator[r_idx[1:-1], c_idx[:-2]] = generator_elements[::2]
        base_operator[r_idx[1:-1], c_idx[2:]] = generator_elements[1::2]
        np.fill_diagonal(base_operator, -np.sum(base_operator, axis=1))

        # -- Boundary Condition: Volatility Matching --
        nabla_0 = self.grid[1] - self.grid[0]
        base_operator[0, 0] = - 1. * self.volatility[0] / (nabla_0 * nabla_0)
        base_operator[0, 1] = - base_operator[0, 0]

        nabla_d = self.grid[self.d - 1] - self.grid[self.d - 2]
        base_operator[self.d - 1, self.d - 1] = - 1. * self.volatility[self.d - 1] / (nabla_d * nabla_d)
        base_operator[self.d - 1, self.d - 2] = - base_operator[self.d - 1, self.d - 1]
        # ----------------------------------------------

        self.sanity_check_base_operator(base_operator)

        return base_operator

    @staticmethod
    def sanity_check_base_operator(base_operator):
        """ Check that rows sum to 1 and every element outside the diagonal is positive
        :param base_operator:
        :return:
        """
        if np.any(np.diag(base_operator) > 0):
            raise ValueError("Error in base generator routine")

        if np.any(abs(base_operator.sum(axis=1)) > 1e-7):
            raise ValueError("Instability in base generator")

    def set_initial_state_index(self, initial_value):
        self.x0 = np.argmin(abs(self.grid - initial_value))

    def plot_probability_distribution_function(self, initial_point=None, title=""):
        import matplotlib.pyplot as plt
        if initial_point is None:
            initial_point = self.x0

        fig = plt.figure()
        n_plots = 1 if self.total_cumulative_stochastic_kernels == {} else 2

        ax1 = fig.add_subplot(1, n_plots, 1)
        if n_plots > 1:
            ax2 = fig.add_subplot(1, n_plots, 2)
        for t in sorted(self.total_stochastic_kernels):
            print("t = " + str(t) + ": " + str(sum(self.total_stochastic_kernels[t][initial_point, :])))
            ax1.plot(self.grid, self.total_stochastic_kernels[t][initial_point, :], label=str(t))
            if n_plots > 1:
                ax2.plot(self.grid, self.total_cumulative_stochastic_kernels[t][initial_point, :],
                         label=str(t))

        ax1.set_title("Probability Distribution Function")
        if n_plots > 1:
            ax2.set_title("Cumulative Distribution Function")

        if n_plots > 1:
            ax2.legend()

        plt.suptitle(title)
        plt.show()
