import os
import unittest
from math import sqrt, exp
from datetime import date, timedelta
import numpy as np

from StochasticModels.LatticeModels.InterestRateModels.CHullWhiteOneFactorModel import HullWhiteOneFactorModel
from NumericalLibrary.COptimiser import Optimiser
from StochasticModels.LatticeModels.EquityModels.CBlackScholesModel import BlackScholesModel
from StochasticModels.ClosedFormModels.CBlackScholesModel import calculate_price
from Curves.CNelsonSiegelDiscountCurve import NelsonSiegelDiscountCurve
from StochasticModels.ClosedFormModels.CHullWhiteModel import discount_factor_observed_at
from StochasticModels.LatticeModels.InterestRateModels.CLogNormalModel import LogNormalModel
from StochasticModels.LatticeModels.InterestRateModels.CShiftedLogNormalModel import ShiftedLogNormalModel


class OptimiserTests(unittest.TestCase):
    """
    Test the uni- and multi-dimensional optimisers, which rely on scipy.optimise
    """

    @staticmethod
    def single_dimensional_merit_function(x):
        return x * x

    @staticmethod
    def multidimensional_merit_function(x_vector):
        return sqrt((x_vector * x_vector).sum()) / len(x_vector)

    def test_brent(self):
        """
        Calculates sqrt(2) numerically by means of Brent method
        :return:
        """
        optimiser = Optimiser(2, 1e-8, 100, self.single_dimensional_merit_function)

        sqrt_2 = optimiser.brent_method(0, 5)

        self.assertTrue(abs(sqrt_2 / sqrt(2) - 1) < 1e-7, "Brent didn't work")

    def test_root_bracketing(self):
        """
        Launch Brent with a non-bracketing interval
        :return:
        """
        optimiser = Optimiser(2, 1e-8, 100, self.single_dimensional_merit_function)

        sqrt_2 = optimiser.brent_method(0, 1)
        self.assertTrue(abs(sqrt_2 / sqrt(2) - 1) < 1e-7, "Monotonic root bracketing didn't work")
        self.assertRaises(Exception, optimiser.brent_method, -1, 1)

    def test_newton(self):
        """
          Calculates sqrt(2) numerically by means of Newton-Raphson method
          :return:
        """
        optimiser = Optimiser(2, 1e-8, 100, self.single_dimensional_merit_function)

        sqrt_2 = optimiser.newton_method(3)

        self.assertTrue(abs(sqrt_2 / sqrt(2) - 1) < 1e-15, "Newton didn't work")

    def test_multidimensional_optimiser(self):
        tolerance = 1e-7

        optimiser = Optimiser(0, tolerance, 1, self.multidimensional_merit_function)

        initial_point_i = [1, 1, 1, 1, 1]
        lower_bound_i = [-1, -2, -3, -4, -5]
        upper_bound_i = [1, 2, 3, 4, 5]

        [_, obj] = optimiser.multidimensional_optimiser(lower_bound_i, upper_bound_i, initial_point_i)

        target_obj = 0
        self.assertTrue(abs(obj - target_obj) < tolerance, "MDO didn't work, OBJ = " + str(obj))
        self.assertTrue(obj < tolerance, "MDO didn't work, OBJ = " + str(obj))

    def test_multidimensional_constrained_optimiser(self):
        tolerance = 1e-7

        def equality_constraint(x):
            # x[0] - 1 = 0
            return x[0] - 1

        def inequality_constraint(x):
            # x_i > 0 for all i
            return x

        optimiser = Optimiser(0, tolerance, 1, self.multidimensional_merit_function)

        initial_point_i = [1, 1, 1, 1, 1]
        lower_bound_i = [-1, -2, -3, -4, -5]
        upper_bound_i = [1, 2, 3, 4, 5]

        [_, obj] = optimiser.multidimensional_optimiser(lower_bound_i, upper_bound_i, initial_point_i,
                                                        inequality_constraint, equality_constraint)

        target_obj = 1.0 / 5
        self.assertTrue(abs(obj - target_obj) < tolerance, "MDO didn't work, OBJ = " + str(obj))


class StochasticModelTests(unittest.TestCase):
    def test_black_scholes(self):
        """
        Verifies that the Lattice Model results match the closed formula up to a given tolerance
        :return:
        """
        x_min = .1
        x_max = 2.5
        d = 128

        n_trials = 4

        moneyness_min = .95
        moneyness_max = 1.0 / moneyness_min
        moneyness_vector = np.linspace(moneyness_min, moneyness_max, n_trials)

        mu_min = 0
        mu_max = .01
        mu_vector = np.linspace(mu_min, mu_max, n_trials)

        sigma_min = sqrt(.01)
        sigma_max = sqrt(.05)
        sigma_vector = np.linspace(sigma_min, sigma_max, n_trials)

        t_vector = [180, 365]

        price_today = 1.5

        for mu in mu_vector:
            for sigma in sigma_vector:
                bs = BlackScholesModel(d, x_min, x_max, mu, sigma)
                bs.create_total_stochastic_kernels(t_vector)
                for moneyness in moneyness_vector:
                    strike = price_today * moneyness
                    bs.set_initial_state_index(price_today)

                    sks = bs.total_stochastic_kernels
                    sk = sks[t_vector[1]]

                    df = exp(-(mu - .5 * sigma * sigma) * 1.0 * t_vector[1] / 365)

                    for position in ["Call", "Put"]:
                        sign = 1 if position == "Call" else -1
                        x0 = bs.x0
                        payoff = np.zeros((d, 1))
                        for x in range(0, d):
                            payoff[x] = max(0, sign * (bs.grid[x] - strike))

                        price_0 = np.dot(sk, payoff)
                        model_price = price_0[x0] * df

                        bs_price = calculate_price(price_today, strike, mu, sigma, t_vector[1], position)

                        self.assertGreaterEqual(bs_price, 0.01,
                                                "Price too low. Should choose another level of moneyness")

                        error = 100 * abs(model_price / bs_price - 1)
                        self.assertLess(error, 3.5)

    def test_hull_white_closed_form(self):
        """
        Verifies that the Lattice Model results match the closed formula up to a given tolerance
        :return:
        """
        valuation_date = date(2015, 1, 1)
        rates_curve = NelsonSiegelDiscountCurve(valuation_date, beta_0=.02, beta_1=-0.0154, beta_2=.01, tau=2)
        observation_date = date(2015, 1, 1)
        days_to_valuation_date = (valuation_date - observation_date).days

        k = 0.01
        sigma = 0.1
        for i in range(1, 100):
            maturity = valuation_date + timedelta(i * 365)
            days_to_maturity = (maturity - observation_date).days

            df = rates_curve.discount_factor(days_to_valuation_date, days_to_maturity)
            df_hw = discount_factor_observed_at(observation_date, valuation_date, maturity, k, sigma, rates_curve)

            error = 10000 * abs(df_hw / df - 1)
            self.assertLess(error, 0.001, "Wrong HW module implementation")

    def test_hull_white_lambda_calibration(self):
        """
        Verifies that the drift adjustment is properly calculated
        :return:
        """
        rates_curve = NelsonSiegelDiscountCurve(date(2000, 1, 1), 2e-2, -2.5e-2, -0.03, .5)

        x_min = -0.01
        x_max = 0.05
        d = 128

        n = 4

        k_min = 0.05
        k_max = 0.15
        k_vec = np.linspace(k_min, k_max, n)

        sigma_min = 0.005
        sigma_max = 0.015
        sigma_vec = np.linspace(sigma_min, sigma_max, n)

        t_min = 60
        t_max = 10000
        t_vector = np.linspace(t_min, t_max, n)

        for k in k_vec:
            for sigma in sigma_vec:
                hw = HullWhiteOneFactorModel(d, x_min, x_max, k, sigma, rates_curve)
                hw.set_initial_state_index(rates_curve.spot_rate(0))
                x0 = hw.x0
                hw.create_total_discounted_stochastic_kernels(t_vector)
                hw.calculate_drift_adjustment()

                for i in range(0, len(t_vector)):
                    discounted_sk = hw.total_discounted_stochastic_kernels[t_vector[i]]
                    adjusted_sk = discounted_sk * hw.total_drift_adjustment[t_vector[i]]
                    model_df = np.dot(adjusted_sk[x0, :], np.ones((d, 1)))
                    market_df = hw.underlying_curve.discount_factor(0, t_vector[i])

                    error = 100 * abs(market_df / model_df - 1)
                    self.assertLess(error, 1e-10, "Calibration didn't work")

    def test_log_normal_lambda_calibration(self):
        """
        Verifies that the drift adjustment is properly calculated
        :return:
        """
        rates_curve = NelsonSiegelDiscountCurve(date(2000, 1, 1), 2e-2, -2.5e-2, -0.03, .5)

        x_min = 0
        x_max = 0.05
        d = 128

        n = 4

        sigma_min = 0.005
        sigma_max = 0.015
        sigma_vec = np.linspace(sigma_min, sigma_max, n)

        t_min = 60
        t_max = 10000
        t_vector = np.linspace(t_min, t_max, n)

        for sigma in sigma_vec:
            ln = LogNormalModel(d, x_min, x_max, sigma, rates_curve)
            ln.set_initial_state_index(rates_curve.spot_rate(0))
            x0 = ln.x0
            ln.create_total_discounted_stochastic_kernels(t_vector)
            ln.calculate_drift_adjustment()

            for i in range(0, len(t_vector)):
                discounted_sk = ln.total_discounted_stochastic_kernels[t_vector[i]]
                adjusted_sk = discounted_sk * ln.total_drift_adjustment[t_vector[i]]
                model_df = np.dot(adjusted_sk[x0, :], np.ones((d, 1)))
                market_df = ln.underlying_curve.discount_factor(0, t_vector[i])

                error = 100 * abs(market_df / model_df - 1)
                self.assertLess(error, 1e-10, "Calibration didn't work")

    def test_shifted_log_normal_lambda_calibration(self):
        """
        Verifies that the drift adjustment is properly calculated
        :return:
        """
        rates_curve = NelsonSiegelDiscountCurve(date(2000, 1, 1), 2e-2, -2.5e-2, -0.03, .5)

        x_min = 0
        x_max = 0.05
        d = 128

        n = 4

        phi_min = 0
        phi_max = 0.04
        phi_vec = np.linspace(phi_min, phi_max, n)

        sigma_min = 0.005
        sigma_max = 0.015
        sigma_vec = np.linspace(sigma_min, sigma_max, n)

        t_min = 60
        t_max = 10000
        t_vector = np.linspace(t_min, t_max, n)

        for phi in phi_vec:
            for sigma in sigma_vec:
                sln = ShiftedLogNormalModel(d, x_min, x_max, phi, sigma, rates_curve)
                sln.set_initial_state_index(rates_curve.spot_rate(0))
                x0 = sln.x0
                sln.create_total_discounted_stochastic_kernels(t_vector)
                sln.calculate_drift_adjustment()

                for i in range(0, len(t_vector)):
                    discounted_sk = sln.total_discounted_stochastic_kernels[t_vector[i]]
                    adjusted_sk = discounted_sk * sln.total_drift_adjustment[t_vector[i]]
                    model_df = np.dot(adjusted_sk[x0, :], np.ones((d, 1)))
                    market_df = sln.underlying_curve.discount_factor(0, t_vector[i])

                    error = 100 * abs(market_df / model_df - 1)
                    self.assertLess(error, 1e-10, "Calibration didn't work")
