#!/bin/python3
import copy


class Opinion:

    def __init__(self, belief: float = 0, disbelief: float = 0, uncertainty: float = 1, base_rate: float = 0) -> None:
        self.set_parameters(belief, disbelief, uncertainty, base_rate)

    def set_parameters(self, belief: float, disbelief: float, uncertainty: float, base_rate: float):
        "Sets opinion parameters and checks their validity"
        # b,d,u should never be touched directly to avoid having invalid opinions
        self._b = belief
        self._d = disbelief
        self._u = uncertainty
        self._a = base_rate
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

    def trust_discounting(self, trust, inplace: bool = False):
        "Discount trust of this opinion according to a trust opinion. Returns the discounted opinion regardless of inplace value"
        discounted = Opinion()
        discounted._b = trust._b * self._b
        discounted._d = trust._b * self._d
        discounted._u = trust._d + trust._u + trust._b * self._u
        discounted._a = self._a

        if inplace:
            self = copy.copy(discounted)

        return discounted

    def projected_probability(self) -> float:
        return self._b + self._a * self._u

    def calculate_conflict(self, reference) -> float:
        "Calculates conflict of this opinion relative to a reference."
        prob_dist = abs(self.projected_probability() -
                        reference.projected_probability())
        conjunctive_conf = (1-reference._u)*(1-self._u)
        return prob_dist * conjunctive_conf

    def __str__(self) -> str:
        return "belief = %g, disbelief = %g, uncertainty = %g, base rate = %g" % (self._b, self._d, self._u, self._a)

    def set_base_rate(self, proba: float) -> None:
        if proba < 0 or proba > 1:
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


def average_fusion(all_opinions: list | tuple) -> Opinion:
    "Calculate the opinion resulting from the average fusion of all opinions passed as argument"
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

    # Calculating ux
    denominator = 0
    for i in range(nb_opinion):
        u_product = _uncertainty_product(all_opinions, i)
        denominator += u_product

    u_product = _uncertainty_product(all_opinions, -1)
    numerator = u_product * nb_opinion
    fused._u = numerator / denominator

    fused._d = 1 - fused._b - fused._u
    fused.validate_opinion()

    return fused


class EBSL:
    "EBSL: Ensemble Binomial Subjective Logic"

    def __init__(self, model, trust_opinion: Opinion) -> None:
        self.model = model
        self.trust_opinion = trust_opinion
        self.trust_offset = 0

    def get_opinion(self, data):
        _, p = self.model.predict_proba(data)
        scale_factor = 1+self.trust_opinion._u
        return Opinion(p/scale_factor, (1-p)/scale_factor, self.trust_opinion._u/scale_factor)
