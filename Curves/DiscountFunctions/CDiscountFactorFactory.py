from Curves.DiscountFunctions.CContinuouslyCompoundedDiscountFactor import ContinuouslyCompoundedDiscountFactor
from Curves.DiscountFunctions.CPeriodicallyCompoundedDiscountFactor import PeriodicallyCompoundedDiscountFactor
from Curves.DiscountFunctions.CLinearDiscountFactor import LinearDiscountFactor


class DiscountFactorFactory:
    def __init__(self, curve_date, day_counter, frequency="inf"):
        self.curve_date = curve_date
        self.day_counter = day_counter

        if ((frequency != "lin" and frequency != "inf") and
                (frequency < 0 or frequency > 12 or (frequency != 0 and 12 % frequency)) != 0):
            raise ValueError("Invalid frequency")
        self.frequency = frequency

    def instantiate(self):
        if self.frequency == "inf":
            discount_factor = ContinuouslyCompoundedDiscountFactor(self.curve_date, self.day_counter)
        elif self.frequency == "lin":
            discount_factor = LinearDiscountFactor(self.curve_date, self.day_counter)
        elif 1 <= self.frequency <= 12:
            discount_factor = PeriodicallyCompoundedDiscountFactor(self.frequency, self.curve_date, self.day_counter)
        else:
            raise NotImplementedError("Unsupported input")

        return discount_factor
