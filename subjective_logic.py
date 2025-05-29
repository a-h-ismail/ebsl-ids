#!/usr/bin/env python3
import copy
import numpy as np


class Opinion:

    def __init__(self, belief: float = 0, disbelief: float = 0, uncertainty: float = 1, base_rate: float = 0) -> None:
        self.set_parameters(belief, disbelief, uncertainty, base_rate)

    def set_parameters(self, belief: float, disbelief: float, uncertainty: float, base_rate: float = -1):
        """Sets opinion parameters and checks their validity. If base rate is not specified, it is unchanged"""
        # b,d,u should never be touched directly to avoid having invalid opinions
        self._b = belief
        self._d = disbelief
        self._u = uncertainty

        if base_rate != -1:
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
            self._b = discounted._b
            self._d = discounted._d
            self._u = discounted._u

        return discounted

    def modify_trust(self, offset: float, inplace=False):
        "Returns the same opinion with b' = b - offset and d' = d + offset"
        # The max positive offset is the belief (because b can't be negative)
        # The min function caps the offset at self._b
        # Reminder: b + d + u = 1
        if offset >= 0:
            offset = min(offset, self._b)
        # Same logic, we get max negative offset = disbelief
        # We use max instead of min here because values are negative
        elif offset < 0:
            offset = max(offset, self._d)

        penalized_trust = copy.copy(self)
        penalized_trust._b -= offset
        penalized_trust._d += offset

        if inplace:
            self._b = penalized_trust._b
            self._d = penalized_trust._d

        return penalized_trust

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

    # Calculate dx
    fused._d = 1 - fused._b - fused._u
    fused.validate_opinion()

    return fused


class BSL_SM:
    "BSLC: Ensemble Binomial Subjective Logic - Single Model"

    def __init__(self, model, trust_opinion: Opinion) -> None:
        self.model = model
        self.trust_opinion = trust_opinion
        self.trust_penalty = 0
        # The opinion is for class 1
        self.information_opinion = Opinion()
        self.conflict = 0
        self.conflict_count = 0

    def get_information_opinion(self, data) -> Opinion:
        """
        Generates the belief opinion of this model based on the prediction probability discounted with initial trust

        Returns:
            opinion: A reference to the internal object storing the opinion
        """
        probabilities = self.model.predict_proba(data)
        self.information_opinion.set_parameters(
            probabilities[1], probabilities[0], 0)
        self.information_opinion.trust_discounting(self.trust_opinion, True)
        return self.information_opinion

    def get_discounted_information_opinion(self) -> Opinion:
        """Calculates the discounted opinion according to the trust opinion modified with trust penalty

        Warning: This function assumes you already called get_information_opinion()

        Returns: A new opinion after trust discounting
        """
        modified_trust = self.trust_opinion.modify_trust(self.trust_penalty)
        return self.information_opinion.trust_discounting(modified_trust)

    def set_prior_probability(self, probability: float):
        self.trust_opinion.set_base_rate(probability)


class EBSL:
    "EBSL: Ensemble Binomial Subjective Logic"

    def __init__(self, conflict_threshold=0.05, max_penalty=0.5, bspeed=1) -> None:
        self.slmodels = []
        self.reference_opinion = Opinion()
        self.conflict_threshold = conflict_threshold
        self.max_penalty = max_penalty
        self.bspeed = bspeed

    def add_model(self, model: BSL_SM):
        self.slmodels.append(model)

    def get_penalty(self, nb_conflict):
        return self.max_penalty*nb_conflict/(nb_conflict + self.bspeed)

    def get_all_opinions_and_ref(self, data):
        "Gets all original model opinions (by inference) then calculates the reference opinion (penalized)"
        discounted_opinions = []
        for slmodel in self.slmodels:
            # Get the information opinion
            opinion = slmodel.get_information_opinion(data)
            # Calculate the trust opinion after penalty
            modified_trust = slmodel.trust_opinion.modify_trust(
                slmodel.trust_penalty)
            # Calculate the information opinion after discounting with penalized trust
            discounted_opinions.append(
                opinion.trust_discounting(modified_trust))

        self.reference_opinion = average_fusion(discounted_opinions)

    def get_all_conflicts(self) -> None:
        for slmodel in self.slmodels:
            slmodel.conflict = slmodel.information_opinion.calculate_conflict(
                self.reference_opinion)

    def reevaluate_trust(self) -> None:
        """Updates the trust for each model according to the conflict"""
        conflicts = [i.conflict for i in self.slmodels]
        average_conflict = np.average(conflicts)
        conflict_offset = np.subtract(average_conflict, conflicts)
        for i in range(len(conflict_offset)):
            if conflict_offset[i] > self.conflict_threshold:
                self.slmodels[i].conflict_count += 1
                self.slmodels[i].trust_penalty = self.get_penalty(
                    self.slmodels[i].conflict_count)

    def get_final_prediction(self) -> float:
        "Calculates the final prediction using discounted information opinion"
        discounted_opinions = []
        for slmodel in self.slmodels:
            discounted_opinions.append(
                slmodel.get_discounted_information_opinion())

        final_opinion = average_fusion(discounted_opinions)
        # The base rate should be the same everywhere
        final_opinion.set_base_rate(discounted_opinions[0]._a)
        prob = final_opinion.projected_probability()

        # Set the base rate for all models to be the newly calculated projected probability
        for slmodel in self.slmodels:
            slmodel.information_opinion.set_base_rate(prob)

        return prob

    def run_once(self, data) -> float:
        # First run inference for each model, estimate opinions and their average
        self.get_all_opinions_and_ref(data)
        self.get_all_conflicts()
        self.reevaluate_trust()
        return self.get_final_prediction()
