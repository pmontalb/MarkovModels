from math import exp


def discount_factor_observed_at(observation_date,
                                valuation_date, maturity, k, sigma, reference_curve):
    """
    :param observation_date:
    :param valuation_date:
    :param maturity:
    :param k: Mean Reversion Rate
    :param sigma:
    :param reference_curve:
    :return:
    """
    days_to_maturity = (maturity - observation_date).days
    days_to_valuation_date = (valuation_date - observation_date).days
    df_0_maturity = reference_curve.discount_factor(0, days_to_maturity)
    df_0_valuation_date = reference_curve.discount_factor(0, days_to_valuation_date)

    deterministic_part = df_0_maturity / df_0_valuation_date

    b_t = calculate_linear_coefficient(valuation_date, k, maturity)
    forward_rate = reference_curve.instantaneous_forward_rate(days_to_valuation_date)
    short_rate = reference_curve.spot_rate(days_to_valuation_date)

    year_fraction = days_to_valuation_date / 365.0
    drift_contribution = (sigma * sigma / (4 * k)) * (1.0 - exp(- 2.0 * k * year_fraction))
    model_contribution = exp(b_t * (forward_rate - drift_contribution * b_t - short_rate))

    df_t_maturity = deterministic_part * model_contribution
    return df_t_maturity


def calculate_linear_coefficient(valuation_date, mean_reversion_rate, maturity):
    """ Using the Affine Term Structure notation, this is the linear coefficient B_t(t, T)
    :param valuation_date:
    :param mean_reversion_rate:
    :param maturity:
    :return:
    """
    year_fraction = (maturity - valuation_date).days / 365
    b_t = 1.0 / mean_reversion_rate * (1.0 - exp(-mean_reversion_rate * year_fraction))
    return b_t


def calculate_mean_reversion_level(days_to_maturity, mean_reversion_rate, sigma, reference_curve):
    """
    :param days_to_maturity:
    :param mean_reversion_rate:
    :param sigma:
    :param reference_curve:
    :return:
    """
    year_fraction = days_to_maturity / 365

    forward_rate = reference_curve.instantaneous_forward_rate(days_to_maturity)
    forward_rate_plus = reference_curve.instantaneous_forward_rate(days_to_maturity + 1.0)
    forward_rate_minus = reference_curve.instantaneous_forward_rate(days_to_maturity - 1.0)

    one_day = 1.0 / 365
    forward_rate_derivative = float(forward_rate_plus - forward_rate_minus) / (2.0 * one_day)

    theta = 0
    theta += forward_rate_derivative
    theta += mean_reversion_rate * forward_rate
    theta += sigma * sigma * 1.0 / (2.0 * mean_reversion_rate) * (1.0 - exp(-mean_reversion_rate * year_fraction))
    return theta
