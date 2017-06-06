from math import exp
from CDiscountCurve import DiscountCurve


class NelsonSiegelDiscountCurve(DiscountCurve):
    def __init__(self, curve_date, beta_0, beta_1, beta_2, tau, frequency="inf"):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        if tau <= 0:
            raise ValueError("Invalid characteristic time")
        self.tau = tau

        super(NelsonSiegelDiscountCurve, self).__init__(curve_date, frequency)

    def spot_rate(self, t):
        if t < 0:
            raise ValueError("t before valuation date")

        t /= 365.0
        characteristic_time = t / self.tau
        exp_char_time = exp(-characteristic_time)

        if t == 0:
            phi_1 = 1
            phi_2 = 0
        else:
            phi_1 = (1 - exp_char_time) / characteristic_time
            phi_2 = phi_1 - exp_char_time

        return self.beta_0 + self.beta_1 * phi_1 + self.beta_2 * phi_2

    def instantaneous_forward_rate(self, t):
        if t < 0:
            raise ValueError("t before valuation date")

        t /= 365.0
        characteristic_time = t / self.tau
        exp_char_time = exp(-characteristic_time)

        inst_fwd = 0
        inst_fwd += self.beta_0
        inst_fwd += self.beta_1 * exp_char_time
        inst_fwd += self.beta_2 * characteristic_time * exp_char_time

        return inst_fwd
