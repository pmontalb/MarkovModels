from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import calculate_d1 as __parent_calculate_d1
from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import calculate_d2 as __parent_calculate_d2
from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import calculate_implied_volatility as \
    __parent_calculate_implied_volatility
from StochasticModels.ClosedFormModels.CGenericOptionBlackScholesModel import \
    calculate_price as __parent_calculate_price


def calculate_d1(underlying, strike, sigma, days_to_maturity):
    """ d1 = (ln(i_F/i_S) + 1/2 * sigma^2 * T) / (sigma * sqrt(T))
    :param underlying:
    :param strike:
    :param sigma:
    :param days_to_maturity:
    :return:
    """
    return __parent_calculate_d1(underlying, strike, 0, 0, sigma, days_to_maturity)


def calculate_d2(underlying, strike, sigma, days_to_maturity):
    """ d2 = d1 - sigma * sqrt(T)
    :param underlying:
    :param strike:
    :param sigma:
    :param days_to_maturity:
    :return:
    """
    return __parent_calculate_d2(underlying, strike, 0, 0, sigma, days_to_maturity)


def calculate_price(forward, strike, annuity, implied_sigma, days_to_maturity, pay_rec):
    """ P_t = A_t * (i_F * N(d_1) - k * N(d_2))
    :param forward:
    :param strike:
    :param annuity:
    :param implied_sigma:
    :param days_to_maturity:
    :param pay_rec:
    :return:
    """
    return __parent_calculate_price(forward, strike, annuity, 0, 0, implied_sigma, days_to_maturity, pay_rec)


def calculate_implied_volatility(forward, strike, annuity, price, days_to_maturity, pay_rec):
    """ Find sigma s.t. P_t = A_t * (i_F * N(d_1(sigma)) - k * N(d_2(sigma))
    :param forward:
    :param strike:
    :param annuity:
    :param price:
    :param days_to_maturity:
    :param pay_rec:
    :return:
    """
    return __parent_calculate_implied_volatility(forward, strike, annuity, 0, 0, price, days_to_maturity, pay_rec)


def calculate_dv01(forward, strike, implied_sigma, annuity, days_to_maturity, pay_rec):
    from scipy.stats import norm
    if pay_rec == "Payer":
        extra_addend = 0
    else:
        if pay_rec == "Receiver":
            extra_addend = -1
        else:
            raise NotImplementedError()

    d1 = calculate_d1(forward, strike, implied_sigma, days_to_maturity)
    dv01 = annuity * (norm.cdf(d1) + extra_addend)

    return dv01


def calculate_vega(forward, strike, implied_sigma, annuity, days_to_maturity):
    """ This function returns Vega, which is the same for both payer/receiver
    :param forward:
    :param strike:
    :param implied_sigma:
    :param annuity:
    :param days_to_maturity:
    :return:
    """
    from scipy.stats import norm
    from math import sqrt
    d1 = calculate_d1(forward, strike, implied_sigma, days_to_maturity)
    year_fraction = float(days_to_maturity) / 365
    vega = annuity * forward * sqrt(year_fraction) * norm.pdf(d1)

    return vega
