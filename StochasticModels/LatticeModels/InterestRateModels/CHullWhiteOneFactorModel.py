from StochasticModels.LatticeModels.InterestRateModels.COneFactorInterestRateModel import OneFactorInterestRateModel


class HullWhiteOneFactorModel(OneFactorInterestRateModel):
    """
    The first two moments for this model are:

        Drift = (0 - k * X)  # Theta is added later on for stability
        Volatility = sigma ^2

    The SDE-equivalent representation is given by:

        dr_t = (theta_t - kappa * r_t) * dt + sigma * dW_t

    Theta is calibrated in such a way it's a step function that matches the underlying curve discount factor. One cannot
    apply the closed formula for the continuous model, meaning that the fit is not perfect at all times, but it's a
    perfect fit for all times in the times vector.
    """
    def __init__(self, d, x_min, x_max, k, sigma, underlying_curve):
        """
        :param d:
        :param x_min:
        :param x_max:
        :param k: Mean Reversion Rate
        :param sigma: Volatility Term
        :return:
        """
        self.k = k
        self.sigma = sigma

        super(HullWhiteOneFactorModel, self).__init__(d, x_min, x_max, underlying_curve)

    def build_moment_vectors(self, t):
        """ Theta is not considered here as it will be applied as a drift adjustment later on.
            In this way the generator is much more stable, and it doesn't depend upon the input reference curve.
        :param t:
        :return:
        """
        from numpy import ones_like

        self.drift = - self.k * self.grid
        self.volatility = (self.sigma * self.sigma) * ones_like(self.grid)
