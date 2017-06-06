from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import calculate_d1 as __parent_calculate_d1
from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import calculate_d2 as __parent_calculate_d2
from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import calculate_implied_volatility as \
    __parent_calculate_implied_volatility
from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import \
    calculate_price as __parent_calculate_price


def calculate_d1(underlying, strike, short_rate, sigma, days_to_maturity):
    """ d1 = (ln(i_F/i_S) + 1/2 * sigma^2 * T) / (sigma * sqrt(T))
    :param short_rate:
    :param underlying:
    :param strike:
    :param sigma:
    :param days_to_maturity:
    :return:
    """
    return __parent_calculate_d1(underlying, strike, short_rate, 0, sigma, days_to_maturity)


def calculate_d2(underlying, strike, short_rate, sigma, days_to_maturity):
    """ d2 = d1 - sigma * sqrt(T)
    :param short_rate:
    :param underlying:
    :param strike:
    :param sigma:
    :param days_to_maturity:
    :return:
    """
    return __parent_calculate_d2(underlying, strike, short_rate, 0, sigma, days_to_maturity)


def calculate_price(underlying, strike, short_rate, implied_sigma, days_to_maturity, pay_rec):
    """ P_t = A_t * (i_F * N(d_1) - k * N(d_2))
    :param short_rate:
    :param underlying:
    :param strike:
    :param implied_sigma:
    :param days_to_maturity:
    :param pay_rec:
    :return:
    """
    return __parent_calculate_price(underlying, strike, 1, short_rate, 0, implied_sigma, days_to_maturity, pay_rec)


def calculate_implied_volatility(forward, strike, short_rate, price, days_to_maturity, pay_rec):
    """ Find sigma s.t. P_t = A_t * (i_F * N(d_1(sigma)) - k * N(d_2(sigma))
    :param short_rate:
    :param forward:
    :param strike:
    :param price:
    :param days_to_maturity:
    :param pay_rec:
    :return:
    """
    return __parent_calculate_implied_volatility(forward, strike, 1, short_rate, 0, price, days_to_maturity, pay_rec)
