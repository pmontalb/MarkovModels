from Curves.DiscountFunctions.CDiscountFactor import DiscountFactor


class LinearDiscountFactor(DiscountFactor):
    def _discount_factor(self, rate, year_fraction):
        return 1.0 / (1 + rate * year_fraction)
