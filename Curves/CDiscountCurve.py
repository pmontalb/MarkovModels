from abc import ABCMeta, abstractmethod
from datetime import date
import matplotlib.pyplot as plt
import numpy as np

from Curves.DiscountFunctions.CDiscountFactorFactory import DiscountFactorFactory


class DiscountCurve(metaclass=ABCMeta):
    def __init__(self, curve_date, frequency="inf"):
        if not isinstance(curve_date, date):
            raise ValueError("Wrong input!")
        self.curve_date = curve_date

        df_factory = DiscountFactorFactory(curve_date,  frequency)
        self.discount_factor_worker = df_factory.instantiate()

    def plot_curves(self, time_horizon=50 * 365):
        """ Plots spot, forward and par rates
        :param time_horizon:
        :return:
        """
        time_vector = np.linspace(0.0, time_horizon, 200)
        spot_rate_vector = [self.spot_rate(s) for s in time_vector]
        forward_rate_vector = [self.instantaneous_forward_rate(s) for s in time_vector]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(time_vector, spot_rate_vector, 'b', label='Spot Rate')
        ax.plot(time_vector, forward_rate_vector, 'r', label='Forward Rate')

        ax.set_xlabel('Time (Days)')
        ax.set_ylabel('Rate')
        plt.legend(loc='best')

        plt.show()

    @abstractmethod
    def spot_rate(self, t):
        """ Return the spot rate observed at time 0 for the period [0, t]
        :param t:
        :return: r_t ( t )
        """
        raise NotImplementedError()

    def discount_factor(self, t0, t1):
        """ Calculates the discount factor for the period [t0, t1]
        :param t0:
        :param t1:
        :return:
        """
        rate_start = self.spot_rate(t0)
        rate_end = self.spot_rate(t1)

        df_end = self.discount_factor_worker.discount_factor(rate_end, t1)
        df_start = self.discount_factor_worker.discount_factor(rate_start, t0)

        return df_end / df_start

    def instantaneous_forward_rate(self, t):
        """ Return the instantaneous forward rate observed at time 0 for the period [t, t + 1]
        :param t: Expressed in days
        :return: f_t (t, t + 1)
        """
        return self.forward_rate(t, t + 1)

    def forward_rate(self, t0, t1):
        """ Returns the arbitrage-free forward rate
        :param t0:
        :param t1:
        :return: (1 + f_t(t0, t1) * (t1 - t0))^(-1) * DF_t(t0) = DF_t(t1), solve for f_t
        """
        if t0 < 0:
            raise ValueError("t0 before valuation date")
        if t1 < 0:
            raise ValueError("t0 before valuation date")
        if t1 <= t0:
            raise ValueError("t1 before than or equal to t0")

        df = self.discount_factor(t0, t1)
        yf = (t1 - t0)  / 365.0

        ratio = 1.0 / df
        fwd_t0_t1 = (ratio - 1) / yf
        return fwd_t0_t1

