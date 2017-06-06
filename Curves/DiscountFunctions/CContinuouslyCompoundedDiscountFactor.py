from Curves.DiscountFunctions.CDiscountFactor import DiscountFactor
from math import exp


class ContinuouslyCompoundedDiscountFactor(DiscountFactor):
    def _discount_factor(self, rate, year_fraction):
        return exp(- rate * year_fraction)
