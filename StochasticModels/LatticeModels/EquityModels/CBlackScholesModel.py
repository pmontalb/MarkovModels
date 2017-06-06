from StochasticModels.LatticeModels.COneFactorModel import OneFactorModel


class BlackScholesModel(OneFactorModel):
    """
    Typical Black-Scholes implementation. The moments are specified as:
        Drift = mu * S
        Volatility = sigma ^ 2 * S ^ 2
    """
    def __init__(self, d, x_min, x_max, mu, sigma):
        """
        :param d:
        :param x_min:
        :param x_max:
        :param mu: Drift Term
        :param sigma: Volatility Term
        :return:
        """
        self.mu = mu
        self.sigma = sigma

        super(BlackScholesModel, self).__init__(d, x_min, x_max)

    def build_moment_vectors(self, t):
        self.drift = self.mu * self.grid
        self.volatility = self.sigma * self.sigma * self.grid * self.grid
