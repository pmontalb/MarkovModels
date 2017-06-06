from Curves.DiscountFunctions.CDiscountFactor import DiscountFactor


class PeriodicallyCompoundedDiscountFactor(DiscountFactor):
    def __init__(self, frequency, curve_date, day_counting_convention):
        self.frequency = frequency
        DiscountFactor.__init__(self, curve_date, day_counting_convention)

    def _discount_factor(self, rate, year_fraction):
        return (1 + 1.0 * rate / self.frequency) ** (-year_fraction * self.frequency)
