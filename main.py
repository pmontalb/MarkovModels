from datetime import date
from Curves.CNelsonSiegelDiscountCurve import NelsonSiegelDiscountCurve
from StochasticModels.LatticeModels.InterestRateModels.CHullWhiteOneFactorModel import HullWhiteOneFactorModel
from StochasticModels.LatticeModels.InterestRateModels.CShiftedLogNormalModel import ShiftedLogNormalModel
from StochasticModels.LatticeModels.InterestRateModels.CShiftedLocalVolatilityModel import ShiftedLocalVolatility
from StochasticModels.LatticeModels.MonteCarloModels.COneFactorInterestRateMonteCarlo import OneFactorInterestRateMonteCarlo
import matplotlib.pyplot as plt
plt.style.use('classic')


def stochastic_model_sk():
    vd = date(2016, 1, 18)
    curve = NelsonSiegelDiscountCurve(vd, 2e-2, -2.5e-2, -0.03, 8.5)

    d = 128
    t = [(i + 1) * 365 for i in range(10)]

    x_min_hw = -.1
    x_max_hw = +.1
    k = .03
    sigma_hw = .012
    hw = HullWhiteOneFactorModel(d, x_min_hw, x_max_hw, k, sigma_hw, curve)

    x_min_sln = 0
    x_max_sln = +.2
    shift = .1
    sigma_sln = .1
    sln = ShiftedLogNormalModel(d, x_min_sln, x_max_sln, shift, sigma_sln, curve)

    x_min_lv = 0.0
    x_max_lv = +.2
    beta = .7
    sigma_lv = .1
    lv = ShiftedLocalVolatility(d, x_min_lv, x_max_lv, shift, sigma_lv, beta, curve)

    hw.set_initial_state_index(hw.underlying_curve.spot_rate(0))
    hw.create_total_stochastic_kernels(t)
    hw.plot_probability_distribution_function(title="Hull-White")

    sln.set_initial_state_index(sln.underlying_curve.spot_rate(0))
    sln.create_total_stochastic_kernels(t)
    sln.plot_probability_distribution_function(title="Shifted Log Normal")

    lv.set_initial_state_index(sln.underlying_curve.spot_rate(0))
    lv.create_total_stochastic_kernels(t)
    lv.plot_probability_distribution_function(title="Local Volatility")


def monte_carlo_simulation():
    vd = date(2016, 1, 18)
    curve = NelsonSiegelDiscountCurve(vd, 2e-2, -2.5e-2, -0.03, .5)

    d = 128
    t = [(i + 1) * 365 for i in range(10)]

    # -------------------------- HULL WHITE ---------------------------------------------------
    x_min_hw = -.1
    x_max_hw = +.1
    k = .03
    sigma_hw = .012
    hw = HullWhiteOneFactorModel(d, x_min_hw, x_max_hw, k, sigma_hw, curve)

    hw_mc = OneFactorInterestRateMonteCarlo(hw, t)
    hw_mc.generate_monte_carlo_path(15)
    hw_mc.plot("Hull White")
    # ------------------------------------------------------------------------------------------

    # ------------------------------- SHIFTED LOG NORMAL ---------------------------------------
    x_min_sln = 0
    x_max_sln = +.2
    phi = .1
    sigma_sln = .1
    sln = ShiftedLogNormalModel(d, x_min_sln, x_max_sln, phi, sigma_sln, curve)

    sln_mc = OneFactorInterestRateMonteCarlo(sln, t)
    sln_mc.generate_monte_carlo_path(15)
    sln_mc.plot("Shifted Log Normal")
    # -------------------------------------------------------------------------------------------

    # ------------------------------- SHIFTED LOG NORMAL ---------------------------------------
    x_min_lv = 0.0
    x_max_lv = +.2
    beta = .7
    sigma_lv = .1
    lv = ShiftedLocalVolatility(d, x_min_lv, x_max_lv, phi, sigma_lv, beta, curve)

    lv_mc = OneFactorInterestRateMonteCarlo(lv, t)
    lv_mc.generate_monte_carlo_path(15)
    lv_mc.plot("Local Volatility")
    # -------------------------------------------------------------------------------------------


if __name__ == "__main__":
    stochastic_model_sk()
