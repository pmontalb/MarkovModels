from math import isnan


class NonMonotonicFunctionException(Exception):
    def __init__(self, message=""):
        super(NonMonotonicFunctionException, self).__init__(message)


class Optimiser:
    def __init__(self, target_value, tolerance, iteration_max, objective_function):
        self.target_value = target_value

        if tolerance <= 0:
            raise ValueError("Invalid tolerance")
        self.tolerance = tolerance

        if iteration_max <= 0:
            raise ValueError("Invalid maximum number of iterations")
        self.iteration_max = iteration_max

        self.objective_function = objective_function

    def objective_function(self, x):
        pass

    def __objective_function(self, x):
        ret = self.objective_function(x) - self.target_value

        if isnan(ret) or ret is None:
            raise ValueError("Objective function is NaN")

        return ret

    def monotonic_root_bracketing(self, a=-1., b=1., search_radius=1.618034, growth_limit=100., max_iterations=1000):
        """
        Bracket the root of a monotonic function.

        Given a monotonic function and two distinct initial points, it adds/subtract a quantity to the upper/lower bound
        until the bracketing condition is reached. It raises a NonMonotonicException if the monotonicity is not verified

        :param a: Initial Lower Bound Estimate
        :param b: Initial Upper Bound Estimate
        :param search_radius: Defines how fast the algorithm converges, and how large the interval [a, b] grows.
                              Should take values in [.5, 2].
        :param growth_limit: maximum variation at every step
        :param max_iterations:
        :return: a, b, f(a), f(b)
        """

        def __monotonicity_check():
            """ Check the mid point in [a, b], and verifies it lies in [f(a), f(b)]
            :return:
            """
            c = .5 * (a + b)
            fc = self.__objective_function(c)

            if fa <= fb and (fc <= fa or fc >= fb):
                raise NonMonotonicFunctionException("f(a) <= f(b) but f(.5 * (a + b)) is not in [f(a), f(b)]")
            if fa >= fb and (fc >= fa or fc <= fb):
                raise NonMonotonicFunctionException("f(a) >= f(b) but f(.5 * (a + b)) is not in [f(b), f(a)]")

        fa = self.__objective_function(a)
        fb = self.__objective_function(b)

        # Swap a and b according to their function values
        if fa > fb:
            a, b = b, a
            fa, fb = fb, fa

        __monotonicity_check()

        n_evaluation = 0

        if fa > 0:
            while fa * fb > 0:

                if n_evaluation >= max_iterations:
                    raise RuntimeError("Failure")

                delta = min(growth_limit, search_radius * (b - a))

                b = a
                fb = fa

                a -= delta
                fa = self.__objective_function(a)

                __monotonicity_check()

                n_evaluation += 1

        else:
            while fa * fb > 0:
                if n_evaluation >= max_iterations:
                    raise RuntimeError("Failure")

                delta = min(growth_limit, search_radius * (b - a))

                a = b
                fa = fb

                b += delta
                fb = self.__objective_function(b)

                __monotonicity_check()

                n_evaluation += 1

        return a, b, fa, fb

    def brent_method(self, lower_bound, upper_bound):
        """
        Brent Method - Inverse Quadratic Interpolation
        Returns a zero x* of the function f in the given interval [a, b], to within a tolerance
        6 * machine_epsilon * |x*| + 2 * tolerance. This function assumes f(a) * f(b) < 0
        :param lower_bound:
        :param upper_bound:
        :return: x* s.t. tgt_value = f(x*)
        """
        from scipy.optimize import brentq

        try:
            x_star = brentq(self.__objective_function, lower_bound, upper_bound,
                            xtol=self.tolerance, maxiter=self.iteration_max)
        except ValueError as err:
            if str(err) != "f(a) and f(b) must have different signs":
                raise
            new_lower_bound, new_upper_bound, _, _ = self.monotonic_root_bracketing(lower_bound, upper_bound)
            x_star = brentq(self.__objective_function, new_lower_bound, new_upper_bound,
                            xtol=self.tolerance, maxiter=self.iteration_max)

        return x_star

    def newton_method(self, initial_point, epsilon=1e-4):
        """ Newton-Raphson method
        :param epsilon: shock parameter for calculating numeric derivatives
        :param initial_point:
        :return: x(n + 1) = x(n) - f(x(n)) / (f'(x(n)), where f'(y) = (f(y + eps) - f(y - eps)) / (2 * eps)
        """

        def f_prime(x):
            f_x_plus = self.__objective_function(x + epsilon)
            f_x_minus = self.__objective_function(x - epsilon)
            f_p = 1.0 * (f_x_plus - f_x_minus) / (2 * epsilon)

            return f_p

        def f_second(x):
            f_x_plus = self.__objective_function(x + epsilon)
            f_x_minus = self.__objective_function(x - epsilon)
            f_x = self.__objective_function(x)
            f_s = 1.0 * (f_x_plus - 2 * f_x + f_x_minus) / (epsilon * epsilon)

            return f_s

        from scipy.optimize import newton
        optimum = newton(self.__objective_function, initial_point,
                         fprime=f_prime, fprime2=f_second,
                         tol=self.tolerance, maxiter=self.iteration_max)

        return optimum

    @staticmethod
    def binary_search(vector, value):
        """ Returns the closest index of vector, x*, such that x* = argmin(|vector[x] - value|)
        :param vector:
        :param value:
        :return:
        """
        import numpy as np

        closest_index_of = np.searchsorted(vector, value)

        return closest_index_of

    def multidimensional_optimiser(self,
                                   lower_bound_i,
                                   upper_bound_i,
                                   initial_point_i,
                                   inequality_constraint_function=None,
                                   equality_constraint_function=None):
        """
        :param lower_bound_i:
        :param upper_bound_i:
        :param initial_point_i:
        :param inequality_constraint_function: callable
        :param equality_constraint_function: callable
        :return:
        """
        from scipy.optimize import minimize

        constraints = []

        if inequality_constraint_function:
            ineq_cons = {'type': 'ineq',
                         'fun': inequality_constraint_function}
            constraints.append(ineq_cons)
        if equality_constraint_function:
            eq_cons = {'type': 'eq',
                       'fun': equality_constraint_function}
            constraints.append(eq_cons)

        bounds = [(lower_bound_i[i], upper_bound_i[i]) for i in range(len(lower_bound_i))]

        optimum = minimize(self.__objective_function, initial_point_i,
                           constraints=constraints, bounds=bounds, options={'disp': False},
                           tol=self.tolerance)

        ret = [optimum.x, optimum.fun]

        return ret
