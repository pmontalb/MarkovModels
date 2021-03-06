from StochasticModels.LatticeModels.InterestRateModels.CLogNormalModel import LogNormalModel


class ShiftedLogNormalModel(LogNormalModel):
    """
    The first two moments for this model are:

        Drift = 0  # mu is added later on for stability
        Volatility = sigma ^2 * r_t^2

    The SDE-equivalent representation is given by:

        d(r_t - phi) = mu_t * (r_t - phi) * dt + sigma * (r_t - phi) * dW_t

    For convenience, the build_grid method will shift the grid in place.

    Mu is calibrated in such a way it's a step function that matches the underlying curve discount factor. One cannot
    apply the closed formula for the continuous model, meaning that the fit is not perfect at all times, but it's a
    perfect fit for all times in the times vector.
    """
    def __init__(self, d, x_min, x_max, phi, sigma, underlying_curve):
        """
        :param d:
        :param x_min:
        :param x_max:
        :param phi: Grid shift
        :param sigma: Volatility Term
        :return:
        """
        if x_min < 0:
            raise ValueError("Log normal model only supports positive rates")

        if x_max < 0:
            raise ValueError("Log normal model only supports positive rates")

        if x_max - phi <= 0:
            raise ValueError("Either Phi is too big or grid upper bound is too low")

        self.phi = phi

        self.sigma = sigma

        super(ShiftedLogNormalModel, self).__init__(d, x_min, x_max, sigma, underlying_curve)

    def build_grid(self, x_min, x_max):
        LogNormalModel.build_grid(self, x_min, x_max)
        self.grid -= self.phi

    def build_moment_vectors(self, t):
        """ Mu is not considered here as it will be applied as a drift adjustment later on.
            In this way the generator is much more stable, and it doesn't depend upon the input reference curve.
        :param t:
        :return:
        """
        # At this stage it's a martingale, but mu is added later on
        self.volatility = self.sigma * self.sigma * (self.grid + self.phi) * (self.grid + self.phi)
