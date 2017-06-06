from CDiscountCurve import DiscountCurve


class FlatRateDiscountCurve(DiscountCurve):
    def __init__(self, flat_rate, curve_date, frequency="inf"):
        self.flat_rate = flat_rate
        super(FlatRateDiscountCurve, self).__init__(curve_date, frequency)

    def spot_rate(self, t):
        return self.flat_rate

    def set_flat_rate(self, flat_rate):
        self.flat_rate = flat_rate
