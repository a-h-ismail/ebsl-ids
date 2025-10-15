#!/bin/env python3
import copy


class Opinion:

    def __init__(self, belief=0., disbelief=0., uncertainty=1., base_rate=0.) -> None:
        self.set_parameters(belief, disbelief, uncertainty, base_rate)

    def set_parameters(self, belief: float, disbelief: float, uncertainty: float, base_rate=-1.):
        """Sets opinion parameters and checks their validity. If base rate is not specified, it is unchanged"""
        # b,d,u should never be touched directly to avoid having invalid opinions
        self._b = belief
        self._d = disbelief
        self._u = uncertainty

        if base_rate != -1:
            self._a = base_rate

        # Do not validate b,d,u,a if the opinion is default initialized
        if belief == 0. and disbelief == 0. and uncertainty == 1. and base_rate == 0.:
            return

        self.validate_opinion()

    def validate_opinion(self):
        if abs(1-self._b-self._d-self._u) > 1e-5:
            raise ValueError(
                "Sum of belief, disbelief and uncertainty should be 1")

        for i in (self._b, self._d, self._u, self._a):
            if i < 0 or i > 1:
                raise ValueError("All parameters should be in range [0;1]")

    def get_parameters(self) -> tuple:
        "Returns (b, d, u, a) of this opinion"
        return (self._b, self._d, self._u, self._a)

    def trust_discounting(self, trust, inplace: bool = False, out=None):
        "Discount trust of this opinion according to a trust opinion. Returns the discounted opinion regardless of inplace value"
        if out is not None:
            discounted = out
        else:
            discounted = Opinion()
        discounted._b = trust._b * self._b
        discounted._d = trust._b * self._d
        discounted._u = 1 - discounted._b - discounted._d
        discounted._a = self._a

        if inplace:
            self._b = discounted._b
            self._d = discounted._d
            self._u = discounted._u

        return discounted

    def modify_trust(self, offset: float, inplace=False, out=None):
        """Returns the same opinion with b' = b - offset and d' = d + offset.
        If "out" is specified, results are written there and inplace is ignored"""
        # -d <= offset <= b
        # Reminder: b + d + u = 1 and b,d >= 0
        if offset > self._b:
            offset = self._b
        elif offset < -self._d:
            # If d becomes 0, u in discounted opinions may become zero, causing problems
            offset = -self._d + 0.05

        if out is not None:
            out._b = self._b - offset
            out._d = self._d + offset
            return out

        if inplace:
            self._b -= offset
            self._d += offset
            return self
        else:
            modified_trust = copy.copy(self)
            if offset == 0:
                return modified_trust
            else:
                modified_trust._b -= offset
                modified_trust._d += offset
                return modified_trust

    def projected_probability(self) -> float:
        return self._b + self._a * self._u

    def calculate_conflict(self, reference) -> float:
        "Calculates conflict of this opinion relative to a reference opinion."
        prob_dist = abs(self.projected_probability() - reference.projected_probability())
        conjunctive_conf = (1-reference._u)*(1-self._u)
        return prob_dist * conjunctive_conf

    def calculate_conflict_fast(self, ref_p, ref_u):
        "Calculates conflict of this opinion relative to a reference opinion. This variant takes precomputed P and u, so it is slightly faster."
        return abs(self.projected_probability() - ref_p) * (1-ref_u)*(1-self._u)

    def __str__(self) -> str:
        return "b = %.3g, d = %.3g, u = %.3g, a = %.3g" % (self._b, self._d, self._u, self._a)

    def copy(self, source):
        """Copies source into this object inplace"""
        self._b = source._b
        self._d = source._d
        self._u = source._u
        self._a = source._a

    def set_base_rate(self, proba: float) -> None:
        # 1.00...1 adds a tiny margin of error accounting for floats errors
        if proba < 0 or proba > 1.000001:
            raise ValueError("Base rate should be in range [0;1]")
        else:
            self._a = proba


def _uncertainty_product(all_opinions: list | tuple, exception_index) -> float:
    "Helper function, calculates the product of uncertainty of all opinions except one"
    product = 1
    for i in range(len(all_opinions)):
        if i == exception_index:
            continue
        else:
            product *= all_opinions[i]._u

    return product


def average_fusion(all_opinions: list | tuple, out=None) -> Opinion:
    "Calculate the opinion resulting from the average fusion of all opinions passed as argument"
    if out is not None:
        fused = out
    else:
        fused = Opinion()

    numerator = 0
    denominator = 0
    nb_opinion = len(all_opinions)
    # Calculating bx
    for i in range(nb_opinion):
        u_product = _uncertainty_product(all_opinions, i)
        numerator += all_opinions[i]._b * u_product
        denominator += u_product
    fused._b = numerator / denominator

    # Calculating ux (reuse denominator calculated above)
    u_product = _uncertainty_product(all_opinions, -1)
    numerator = u_product * nb_opinion
    fused._u = numerator / denominator

    # Calculate dx
    fused._d = 1 - fused._b - fused._u

    return fused


def average_fusion_demo(all_opinions: list | tuple):
    # Add some spacing after any previous content
    print("")
    length = len(all_opinions)
    # Get the real average from the proper function
    avg_opinion = average_fusion(all_opinions)
    # Show how b is calculated
    # The numerator first
    numerator = ""
    for i in range(length):
        first = True
        numerator += str(all_opinions[i]._b)
        # add the uncertainty product
        for j in range(length):
            if j == i:
                continue
            else:
                if first:
                    numerator += " * (" + str(all_opinions[j]._u)
                    first = False
                else:
                    numerator += " * " + str(all_opinions[j]._u)
        numerator += ")"
        # If not at the last element, add a + sign
        if i < length - 1:
            numerator += " + "
    numerator = numerator.lstrip()

    # The denominator of b
    denominator = ""
    for i in range(length):
        first = True
        for j in range(length):
            if j == i:
                continue
            else:
                if first:
                    denominator += "(" + str(all_opinions[j]._u)
                    first = False
                else:
                    denominator += " * " + str(all_opinions[j]._u)
        denominator += ") "
        if i < length - 1:
            denominator += "+ "
    denominator = denominator.rstrip()
    # Display b
    print("   ", numerator)
    print("b =", "—"*len(numerator), "=", "%.4g" % avg_opinion._b)
    print(" " * ((len(numerator)-len(denominator))//2 + 3), denominator)

    # Now create the numerator for u
    first = True
    numerator = str(length)
    for j in range(length):
        if first:
            numerator += " * (" + str(all_opinions[j]._u)
            first = False
        else:
            numerator += " * " + str(all_opinions[j]._u)
    numerator += ")"

    # d is straightforward
    print("\nd = 1 - b - u = %.4g" % avg_opinion._d)

    # Printing u
    print("")
    print(" " * ((len(denominator)-len(numerator))//2 + 3), numerator)
    print("u =", "—"*len(denominator), "=", "%.4g" % avg_opinion._u)
    print("   ", denominator)

    print("\nAverage Opinion: w_avg = (%s)" % avg_opinion)
