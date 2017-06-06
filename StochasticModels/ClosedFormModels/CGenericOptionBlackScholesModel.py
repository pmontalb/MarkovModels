""""
This is the most generic implementation of the BS model: its usage are 3:
    - Black-Scholes for Equity: r_f == 0, annuity == 1
    - Black-76 for Swaption: r_f == 0
    - Black-Scholes for FX: annuity == 1
"""


def calculate_d1(underlying, strike, domestic_short_rate, foreign_short_rate, sigma, days_to_maturity):
    """ d1 = (ln(i_F/i_S) + ((r_d - r_f) + 1/2 * sigma^2) * T) / (sigma * sqrt(T))
    :param domestic_short_rate:
    :param foreign_short_rate:
    :param underlying:
    :param strike:
    :param sigma:
    :param days_to_maturity:
    :return:
    """
    from math import log, sqrt
    if days_to_maturity < 0:
        raise ValueError("Invalid days to maturity")
    year_fraction = float(days_to_maturity) / 365

    if underlying < 0:
        raise ValueError("Invalid underlying")
    if strike <= 0:
        raise ValueError("Invalid Strike")

    moneyness = float(underlying) / strike
    log_moneyness = log(moneyness)
    r = domestic_short_rate - foreign_short_rate

    numerator = log_moneyness + (r + .5 * sigma * sigma) * year_fraction
    denominator = sigma * sqrt(year_fraction)

    d1 = float(numerator) / denominator

    return d1


def calculate_d2(underlying, strike, domestic_short_rate, foreign_short_rate, sigma, days_to_maturity):
    """ d2 = d1 - sigma * sqrt(T)
    :param domestic_short_rate:
    :param foreign_short_rate:
    :param underlying:
    :param strike:
    :param sigma:
    :param days_to_maturity:
    :return:
    """
    from math import sqrt
    year_fraction = float(days_to_maturity) / 365
    d1 = calculate_d1(underlying, strike, domestic_short_rate, foreign_short_rate, sigma, days_to_maturity)
    return d1 - sigma * sqrt(year_fraction)


def __calculate_price(underlying, strike, annuity, domestic_short_rate, foreign_short_rate,
                      implied_sigma, days_to_maturity, sign):
    """ P_t = A_t * (S_t * N(d_1) - k * e^(- (r_d - r_f) * T) * N(d_2))
    :param underlying:
    :param strike:
    :param annuity:
    :param implied_sigma:
    :param days_to_maturity:
    :param sign:
    :return:
    """
    from scipy.stats import norm
    from math import exp

    if sign != 1 and sign != -1:
        raise ValueError("Invalid Sign")

    d1 = calculate_d1(underlying, strike, domestic_short_rate, foreign_short_rate, implied_sigma, days_to_maturity)
    d2 = calculate_d2(underlying, strike, domestic_short_rate, foreign_short_rate, implied_sigma, days_to_maturity)

    r = domestic_short_rate - foreign_short_rate
    year_fraction = float(days_to_maturity) / 365
    df = exp(-r * year_fraction)
    price = annuity * sign * (underlying * norm.cdf(sign * d1) - strike * df * norm.cdf(sign * d2))

    return price


def calculate_price(underlying, strike, annuity, domestic_short_rate, foreign_short_rate,
                    implied_sigma, days_to_maturity, pay_rec):
    """ P_t = A_t * (S_t * N(d_1) - k * e^(- (r_d - r_f) * T) * N(d_2))
    :param domestic_short_rate:
    :param foreign_short_rate:
    :param underlying:
    :param strike:
    :param annuity:
    :param implied_sigma:
    :param days_to_maturity:
    :param pay_rec:
    :return:
    """
    if pay_rec == "Payer" or pay_rec == "Call":
        sign = 1
    else:
        if pay_rec == "Receiver" or pay_rec == "Put":
            sign = -1
        else:
            raise NotImplementedError()

    return __calculate_price(underlying, strike, annuity, domestic_short_rate, foreign_short_rate,
                             implied_sigma, days_to_maturity, sign)


def calculate_implied_volatility(underlying, strike, annuity, domestic_short_rate, foreign_short_rate,
                                 price, days_to_maturity, pay_rec):
    """ Find sigma s.t. P_t = A_t * (S_t * N(d_1(sigma)) - k * e^(- (r_d - r_f) * T) * N(d_2(sigma)))
    :param domestic_short_rate:
    :param foreign_short_rate:
    :param underlying:
    :param strike:
    :param annuity:
    :param price:
    :param days_to_maturity:
    :param pay_rec:
    :return:
    """
    from NumericalLibrary.COptimiser import Optimiser
    from math import exp

    sigma_min = 0.0001
    sigma_max = 10

    tolerance = 1e-7
    iteration_max = 100

    # --- Sanity checks --- #
    r = domestic_short_rate - foreign_short_rate
    year_fraction = float(days_to_maturity) / 365
    df = exp(-r * year_fraction)

    price_lb = 1e9
    price_ub = 0

    if pay_rec == "Payer" or pay_rec == "Call":
        price_lb = annuity * (underlying - strike * df)
        price_ub = annuity * underlying
    else:
        if pay_rec == "Receiver" or pay_rec == "Put":
            price_lb = annuity * (strike - underlying)
            price_ub = annuity * strike

    # P_t / A_t \in [i_s - i_k, i_s]
    if price < price_lb or price > price_ub:
        raise ValueError("Price not allowed in the Black model")

    if abs(price - price_lb) <= tolerance:
        raise ValueError("Price hits its lower bound. Impossible to calculate the implied vol")

    if abs(price - price_ub) <= tolerance:
        raise ValueError("Price hits its upper bound. Impossible to calculate the implied vol")

    # ---  --- #

    def internal_calculate_price(sigma):
        return calculate_price(underlying, strike, annuity, domestic_short_rate, foreign_short_rate,
                               sigma, days_to_maturity, pay_rec)

    optimiser = Optimiser(price, tolerance, iteration_max, internal_calculate_price)

    implied_vol = optimiser.brent_method(sigma_min, sigma_max)

    if abs(implied_vol - sigma_min) <= tolerance:
        raise ValueError("Hit lower bound. Algorithm might have not converged")
    if abs(implied_vol - sigma_max) <= tolerance:
        raise ValueError("Hit upper bound. Algorithm might have not converged")

    return implied_vol

