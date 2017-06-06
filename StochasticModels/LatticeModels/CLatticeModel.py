from abc import abstractmethod, ABCMeta
import numpy as np
import scipy.linalg as linalg


class LatticeModel(metaclass=ABCMeta):
    """
    This class is the base class for all Lattice Models.

    Thanks to having a defined lattice, is possible to price whatever instrument with no closed formula.

    The model is fully defined when the local drift and the local volatility are specified
    (e.g. implement "build_moment_vectors").

    As for a trinomial tree, the main player here is the probability of going from an initial state to a final state in
    a given time. Since we use a pre-defined grid, it is possible to write these probability in the form of a matrix.

    Note that any noticeable pattern (tri-diagonal, block tri-diagonal) depends on how you write your state variable
    vector. The current implementation supports only a
    One-Factor model that leads to a tri-diagonal elementary matrix.

    The name convention used here is the following:

        - Initial Condition: as in a tree you have only 1 point at the beginning, your initial state vector is a vector
                             u_0 = (0, 0, ..., 0, 1, 0, ..., 0, 0) where 1 is in correspondence of the initial grid
                             point.

        - Base Generator: matrix that represents the rate of probability in an infinitesimal time dt
                          (usually set as low as 1 / 365, e.g. 1 day). For fully defining an Elementary Generator
                           you only need a drift vector and a volatility vector. Row sum needs to be 0 for
                           probability conservation.

        - Stochastic Kernel (SK): matrix that represent the transition probabilities from one grid point to
                                  another in a given time period. In this way, given a time period dt, I
                                  can go from dt to 2 * dt just multiplying two kernels such as:
                                      u_{2 * dt} = u_{dt} \cdot u_{dt}

    Here all the model parameters are thought to be constant between a certain time vector items. This time vector
    usually coincides with the cash flow vector. It's not possible to discount a cash flow if a SK has not been
    calculated for discounting from that time to 0.

    Note that once a times vector is used it --CANNOT-- change during the calculations, otherwise an error is triggered.

    The attribute "partial" refers to calculating the expected value from a given time to its --PREDECESSOR-- in
    the times vector.

    The attribute "total" refers to calculating the expected value from a given time to 0.

    """
    def __init__(self, d):
        """
        :param d: d = nx1 * nx2 * ... nxM, where M is the number of factors
        :return:
        """
        self.d = d
        self.grid = np.zeros(d)

        self.total_drift_adjustment = {}

        self.stochastic_kernels = {}
        self.total_stochastic_kernels = {}

        self.total_cumulative_stochastic_kernels = {}

    def build_base_operator(self, t):
        """ Build the Markov operator L at time t
        :param t:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def build_moment_vectors(self, t):
        """ Build drift and volatility vectors
        :param t:
        :return:
        """
        raise NotImplementedError()

    def create_markov_operator(self, base_operator, dt, method="CrankNicolson"):
        """ Let u be the Markov operator and L be the base generator. The model dynamic
            is given by
                            u ' = L \cdot u
            Following the Method Of Line (MOL) discretization,
            let u(n) = u(t_n) where t_n is the discretized time domain.

            The following schemes can be used:
            - Explicit Euler --> u(n + 1) = (I + L * dt) \cdot u(n)
            - Implicit Euler --> (I - L * dt) \cdot u(n + 1) = u(n)
            - Crank Nicolson --> (I - L * dt / 2) \cdot u(n + 1) = (I + L * dt / 2) \cdot u(n)
        :param base_operator:
        :param dt:
        :param method:
        :return:
        """
        if method == "ExplicitEuler":
            u = np.eye(self.d) + dt * base_operator
        elif method == "ImplicitEuler":
            u = linalg.solve(np.eye(self.d) - dt * base_operator, np.eye(self.d))
        elif method == "CrankNicolson":
            u = linalg.solve(np.eye(self.d) - .5 * dt * base_operator,
                             np.eye(self.d) + .5 * dt * base_operator)
        else:
            raise NotImplementedError("Unsupported SDE resolution method")
        return u

    def create_stochastic_kernels(self, t_vector, solver="CrankNicolson"):
        """ Creates the sk for [t_{i - 1}, t_i] for each t_i in t_vec.
            Must be called for pricing by means of backward/forward induction
        :param t_vector: Vector of days from valuation date
        :param solver:
        :return:
        """
        from math import log
        current_sk = self.get_current_stochastic_kernels(is_total_transition=False)

        def __worker():
            t = t_vector[i]
            t_minus_1 = t_vector[i - 1] if i > 0 else 0

            if t == 0:
                raise ValueError("t must be greater than 0")

            base_operator = self.build_base_operator(t)

            if solver == "ExplicitEuler":
                dt = .95 / (np.linalg.norm(base_operator, np.inf))
            else:
                # Not needed for stability, but good for consistency
                dt = 1.0 / (np.linalg.norm(base_operator, np.inf))

            year_fraction = (t - t_minus_1) / 365.0

            # -- Operator Exponentiation --
            n_exponentiation = int(log(year_fraction / dt, 2)) + 1
            dt = year_fraction * (2 ** (-n_exponentiation))

            markov_operator = self.create_markov_operator(base_operator, dt, solver)
            self.sanity_check_sk(markov_operator)

            for _ in range(n_exponentiation):
                markov_operator = markov_operator.dot(markov_operator)
            # -------------------------

            current_sk[t] = markov_operator
            self.sanity_check_sk(current_sk[t])

        for i in range(len(t_vector)):
            __worker()

    def create_total_stochastic_kernels(self, t_vector, solver="CrankNicolson"):
        """ Must be called for pricing by means of Backward Induction
        :param t_vector:
        :param solver:
        :return:
        """
        self.create_stochastic_kernels(t_vector, solver)

        def __worker():
            current_sk = self.get_current_stochastic_kernels(is_total_transition=False)
            current_total_sk = self.get_current_stochastic_kernels(is_total_transition=True)

            if i == 0:
                total_sk = current_sk[t_vector[i]]
            else:
                total_sk = current_total_sk[t_vector[i - 1]].dot(current_sk[t_vector[i]])

            current_total_sk[t_vector[i]] = total_sk

        for i in range(len(t_vector)):
            __worker()

    def create_total_cumulative_stochastic_kernels(self):
        if self.total_stochastic_kernels == {}:
            raise RuntimeError("First create the SKs")

        for t, sk in self.total_stochastic_kernels.items():
            self.total_cumulative_stochastic_kernels[t] = self.__accumulate_sk(sk)

    def __accumulate_sk(self, sk):
        cumulative_sk = np.zeros_like(sk)

        for x in range(self.d):
            accumulator = 0.
            for y in range(self.d):
                accumulator += sk[x, y]
                cumulative_sk[x, y] = accumulator

            if abs(cumulative_sk[x, -1] - 1) > 1e-10:
                raise ValueError("Kernel doesn't sum up to 1")

        return cumulative_sk

    def get_times_vector(self):
        """ Returns the times at which the SKs are calculated
        :return:
        """
        keys = sorted(self.stochastic_kernels)
        check_keys = sorted(self.total_stochastic_kernels)

        if check_keys != keys:
            raise ValueError("Total SK and SK are calculated at different times!")

        return keys

    def sanity_check_sk(self, sk):
        """ Check unitary row sum and boundary check (in [0, 1] )
        :param sk:
        :return:
        """
        if np.any(sk < 0) and np.any(sk > 1):
            raise ValueError("Instability")

        if np.any(abs(sk.sum(axis=1) - 1) > 1e-10):
            raise ValueError("Instability in FEX")

    def get_current_stochastic_kernels(self, is_total_transition):
        return self.total_stochastic_kernels if is_total_transition else self.stochastic_kernels

    def clear_all_stochastic_kernels(self):
        self.total_stochastic_kernels.clear()
        self.stochastic_kernels.clear()
