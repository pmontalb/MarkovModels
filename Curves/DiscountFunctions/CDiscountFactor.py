class DiscountFactor:
    def __init__(self, curve_date, day_counter):
        self.curve_date = curve_date
        self.day_counter = day_counter

    def calculate_year_fraction(self, t):
        """ Return the discount factor using continuous compounding
        :param t:
        :return: e ^ [-r_0 (t) * year_fraction(0 , t)]
        """
        try:
            return t / 365.
        except (TypeError, AttributeError):
            return self.day_counter.year_fraction(self.curve_date, t)

    def discount_factor(self, rate, time):
        yf = self.calculate_year_fraction(time)
        return self._discount_factor(rate, yf)

    def _discount_factor(self, rate, yf):
        raise NotImplementedError()
