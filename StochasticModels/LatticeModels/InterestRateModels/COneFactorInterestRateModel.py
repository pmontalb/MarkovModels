from StochasticModels.LatticeModels.COneFactorModel import OneFactorModel
from functools import reduce


class OneFactorInterestRateModel(OneFactorModel):
    """
    When using an Interest Rate model is much more convenient to enclose the discount factor calculation inside the SK
    routine.

    One then has an "un-discounted" SK, which serves to calculate a pure expected value, and a "discounted" SK, that
    calculates discounted expectations.

    Usually, one needs both.
    """
    def __init__(self, d, x_min, x_max, underlying_curve):
        self.__build_discounted_stochastic_kernels = False

        self.total_discounted_stochastic_kernels = {}
        self.discounted_stochastic_kernels = {}

        self.drift_adjustment = {}  # market price of risk

        self.underlying_curve = underlying_curve

        super(OneFactorInterestRateModel, self).__init__(d, x_min, x_max)

    def __discount_markov_operator(self, u, dt):
        if not self.__build_discounted_stochastic_kernels:
            raise RuntimeError("Wrong call! If this flag is not active, this method shouldn't be called!")
        from numpy import exp, newaxis
        u *= exp(- self.grid[:, newaxis] * dt)

        return u

    def create_markov_operator(self, base_operator, dt, solver="CrankNicolson"):
        """ Returns the discounted Markov operator. Kernels obtained with this are discounted kernels.
        :param base_operator:
        :param dt:
        :param solver:
        :return:
        """
        u = super(OneFactorInterestRateModel, self).create_markov_operator(base_operator, dt, solver)
        super(OneFactorInterestRateModel, self).sanity_check_sk(u)

        if self.__build_discounted_stochastic_kernels:
            u = self.__discount_markov_operator(u, dt)
        return u

    def create_discounted_stochastic_kernel(self, t_vector, solver="CrankNicolson"):
        self.__build_discounted_stochastic_kernels = True
        super(OneFactorInterestRateModel, self).create_stochastic_kernels(t_vector, solver)
        self.__build_discounted_stochastic_kernels = False

    def create_total_discounted_stochastic_kernels(self, t_vector, solver="CrankNicolson"):
        self.__build_discounted_stochastic_kernels = True
        super(OneFactorInterestRateModel, self).create_total_stochastic_kernels(t_vector, solver)
        self.__build_discounted_stochastic_kernels = False

    def get_current_stochastic_kernels(self, is_total_transition):
        if self.__build_discounted_stochastic_kernels:
            if is_total_transition:
                return self.total_discounted_stochastic_kernels
            else:
                return self.discounted_stochastic_kernels

        return super(OneFactorInterestRateModel, self).get_current_stochastic_kernels(is_total_transition)

    def sanity_check_sk(self, sk):
        """ No checks on Discounted Transition Probability Kernels are possible...
        :param sk:
        :return:
        """
        if self.__build_discounted_stochastic_kernels:
            return
        else:
            super(OneFactorInterestRateModel, self).sanity_check_sk(sk)

    def clear_all_stochastic_kernels(self):
        self.discounted_stochastic_kernels.clear()
        self.total_discounted_stochastic_kernels.clear()
        super(OneFactorInterestRateModel, self).clear_all_stochastic_kernels()

    def get_times_vector(self):
        keys = super(OneFactorInterestRateModel, self).get_times_vector()
        if len(keys) == 0:
            keys = sorted(self.discounted_stochastic_kernels)
            check_keys = sorted(self.total_discounted_stochastic_kernels)
            if keys != check_keys:
                raise ValueError("Total SK and SK are calculated at different times!")

        if len(keys) == 0:
            raise ValueError("Model hasn't been calibrated!")

        return keys

    def calculate_drift_adjustment(self):
        """ This function finds lambda at time t such that the model discount factor in [0, t] matches the market one.
            For a Lattice Model, it coincides with an upper triangular matrix, as the adjustment in [t_i, t_j] is
            conditioned to the filtration in t_i.

            Since in this framework we model the unadjusted rate, r*,  it holds:
                r(t) = \int lambda(s)ds + r*(t)

            Passing to Discount Factors:
                DF(s, t) = exp( - \int lambda_s(u) du) * DF*(s, t)

            But the LHS is the market DF, and the last factor in RHS is the x0-th row of the unadjusted discounted SK
            times the payoff vector of ones.

            Taking logarithm at both sides takes to:
                \int lambda_s(u) du = - ln( DF_MKT(s, t) / DF*(s, t))

            Since we assume a piecewise constant lambda, the integral becomes a sum:
                \sum lambda_i * \tau_i = rho_i,
            where rho_i is the log of the DF ratio.

            This leads to a lower triangular linear system in theta_i:

                lambda_s(0) * tau_0                                                           = rho_0
                lambda_s(0) * tau_0 + lambda_s(1) * rho_1                                     = rho_1

                .
                .
                .

                lambda_s(0) * tau_0 + ...                        ... + lambda_s(m) * tau_m    = rho_m

            Denoting tau_matrix the year fraction lower triangular matrix, and rho_vector the RHS, theta is the solution
            of:
                tau_matrix \cdot theta_vector_s = rho_vector

            Finally, we need to apply this adjustment to both total/partial discounted SKs

        :return:
        """
        import numpy as np
        from math import exp
        import scipy.linalg as linalg

        t_vector = self.get_times_vector()

        self.set_initial_state_index(self.underlying_curve.spot_rate(0))

        payoff = np.ones(self.d)
        m = len(t_vector)

        tau_vector = np.zeros(m)
        tau_vector[0] = 1. / 365 * t_vector[0]
        tau_vector[1:] = 1. / 365 * np.diff(t_vector)

        tau_matrix = np.vstack([tau_vector] * m)
        tau_matrix[np.triu_indices(m, 1)] = 0.

        def vectorised_market_df(t):
            return self.underlying_curve.discount_factor(0, t)

        def vectorised_model_df(t):
            return np.dot(self.total_discounted_stochastic_kernels[t][self.x0, :], payoff)

        market_df_vector = np.vectorize(vectorised_market_df)(t_vector)
        model_df_vector = np.vectorize(vectorised_model_df)(t_vector)
        rho_vector = - np.log(market_df_vector / model_df_vector)

        lambda_vector = linalg.solve(tau_matrix, rho_vector)

        # --- Only for displaying purposes ---
        self.drift_adjustment = {t_vector[i]: lambda_vector[i] for i in range(m)}
        # -------------------------------------------

        # Cache the drift adjustments
        self.total_drift_adjustment = {t_vector[i]: exp(- np.dot(lambda_vector, tau_matrix[i, :])) for i in range(m)}

    def get_stochastic_kernel(self, start_days, end_days):
        """ Returns the sk correspondent to the period [start, end].
            If start = 0, returns the x0-th row of the total sk in [0, end]
        :param start_days:
        :param end_days:
        :return:
        """
        import numpy as np

        if end_days == 0 and start_days == 0:
            return [1]

        if start_days == 0:
            return self.total_stochastic_kernels[end_days][self.x0, :]
        else:
            sk_iterator = (v for k, v in self.stochastic_kernels.items()
                            if start_days < k <= end_days)
            cumulative_sk = reduce(np.dot, sk_iterator)

            return cumulative_sk

    def get_discounted_stochastic_kernel(self, start_days, end_days):
        """ Returns the sk correspondent to the period [start, end].
            If start = 0, returns the x0-th row of the total sk in [0, end]
        :param start_days:
        :param end_days:
        :return:
        """
        import numpy as np

        if end_days == 0 and start_days == 0:
            return [1]

        if start_days == 0:
            sk = self.total_discounted_stochastic_kernels[end_days][self.x0, :]
            return sk * self.total_drift_adjustment[end_days]
        else:
            sk_iterator = (v for k, v in self.discounted_stochastic_kernels.items()
                            if start_days < k <= end_days)
            cumulative_sk = reduce(np.dot, sk_iterator)

            return cumulative_sk * (self.total_drift_adjustment[end_days] / self.total_drift_adjustment[start_days])

