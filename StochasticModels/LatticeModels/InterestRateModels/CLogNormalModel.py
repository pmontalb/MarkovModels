from StochasticModels.LatticeModels.InterestRateModels.COneFactorInterestRateModel import OneFactorInterestRateModel


class LogNormalModel(OneFactorInterestRateModel):
    """
    The first two moments for this model are:

        Drift = 0  # mu is added later on for stability
        Volatility = sigma ^2 * r_t^2

    The SDE-equivalent representation is given by:

        dr_t = mu_t * r_t * dt + sigma * r_t * dW_t

    Mu is calibrated in such a way it's a step function that matches the underlying curve discount factor. One cannot
    apply the closed formula for the continuous model, meaning that the fit is not perfect at all times, but it's a
    perfect fit for all times in the times vector.
    """

    def __init__(self, d, x_min, x_max, sigma, underlying_curve):
        """
        :param d:
        :param x_min:
        :param x_max:
        :param sigma: Volatility Term
        :return:
        """
        if x_min < 0:
            raise ValueError("Log normal model only supports positive rates")

        if x_max < 0:
            raise ValueError("Log normal model only supports positive rates")

        self.sigma = sigma

        super(LogNormalModel, self).__init__(d, x_min, x_max, underlying_curve)

    def build_moment_vectors(self, t):
        """ Mu is not considered here as it will be applied as a drift adjustment later on.
            In this way the generator is much more stable, and it doesn't depend upon the input reference curve.
        :param t:
        :return:
        """
        # At this stage it's a martingale, but drift is added later on
        self.volatility = self.sigma * self.sigma * self.grid * self.grid
