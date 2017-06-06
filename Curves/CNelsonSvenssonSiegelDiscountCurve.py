from math import exp
from CDiscountCurve import DiscountCurve


class NelsonSvenssonSiegelDiscountCurve(DiscountCurve):
    def __init__(self, curve_date, beta_0, beta_1, beta_2, beta_3, tau, nu, frequency="inf"):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.beta_3 = beta_3

        if tau <= 0:
            raise ValueError("Invalid first characteristic time")
        self.tau = tau

        if nu <= 0:
            raise ValueError("Invalid second characteristic time")
        self.nu = nu

        super(NelsonSvenssonSiegelDiscountCurve, self).__init__(curve_date, frequency)

    def spot_rate(self, t):
        if t < 0:
            raise ValueError("t before valuation date")

        t /= 365.0
        first_characteristic_time = t / self.tau
        exp_first_char_time = exp(-first_characteristic_time)
        second_characteristic_time = t / self.nu
        exp_second_char_time = exp(-second_characteristic_time)

        if t == 0:
            phi_1 = 1
            phi_2 = 0
            phi_3 = 0
        else:
            phi_1 = (1 - exp_first_char_time) / first_characteristic_time
            phi_2 = phi_1 - exp_first_char_time
            phi_3 = (1 - exp_second_char_time) / second_characteristic_time - exp_second_char_time

        rate = self.beta_0 + self.beta_1 * phi_1 + self.beta_2 * phi_2 + self.beta_3 * phi_3
        return rate

    def instantaneous_forward_rate(self, t):
        if t < 0:
            raise ValueError("t before valuation date")

        t /= 365.0

        first_characteristic_time = t / self.tau
        exp_first_char_time = exp(-first_characteristic_time)

        second_characteristic_time = t / self.nu
        exp_second_char_time = exp(-second_characteristic_time)

        inst_fwd = 0
        inst_fwd += self.beta_0
        inst_fwd += self.beta_1 * exp_first_char_time
        inst_fwd += self.beta_2 * first_characteristic_time * exp_first_char_time
        inst_fwd += self.beta_3 * second_characteristic_time * exp_second_char_time

        return inst_fwd
