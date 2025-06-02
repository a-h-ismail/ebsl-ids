#!/usr/bin/env python3
import copy
from typing import Literal
import numpy as np


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
        return "b = %g, d = %g, u = %g, a = %g" % (self._b, self._d, self._u, self._a)

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


def find_max_belief(all_opinions: list | tuple):
    "Returns the index of the highest belief in the provided opinions"
    max = -1
    max_index = -1
    for i in range(len(all_opinions)):
        if max < all_opinions[i]._b:
            max_index = i
            max = all_opinions[i]._b

    return max_index


class BSL_SM:
    "BSL_SM: Binomial Subjective Logic - Single Model"

    def __init__(self, model, scaler, trust_opinion: Opinion) -> None:
        """
        Creates the building blocks required for Ensemble Binomial Subjective Logic.
        It encapsulates an ML model, its scaler and manages subjective logic opinions (information, trust)

        Parameters
        ----------
        model: Any binary model providing methods predict() and predict_proba() like sklearn

        scaler: Should provide the transform() method like sklearn

        trust_opinion: Indicates the trustworthiness of a model. Affects the contribution of each model to the final prediction.
        """
        self.model = model
        self.scaler = scaler
        self.trust_opinion = trust_opinion
        self.penalized_trust_opinion = copy.copy(trust_opinion)
        self.trust_penalty = 0.
        # The opinion is for class 1
        self.information_opinion = Opinion()
        self.conflict = 0.
        self.conflict_count = 0

    def predict_proba(self, samples) -> np.ndarray:
        """
        Calls predict_proba of the underlying model after scaling the input samples
        """
        if self.scaler is not None:
            samples = self.scaler.transform(samples)
        return self.model.predict_proba(samples)

    def get_information_opinion(self, samples):
        """Generates the belief opinion of this model based on the prediction probability"""
        probabilities = self.predict_proba(samples)[0]
        self.information_opinion.set_parameters(
            probabilities[1], probabilities[0], 0)

    def get_discounted_information_opinion(self) -> Opinion:
        """Calculates the discounted opinion according to the trust opinion modified with trust penalty

        Warning: This function assumes you already called get_information_opinion()

        Returns: A new opinion after trust discounting (original opinion unchanged)
        """
        modified_trust = self.trust_opinion.modify_trust(self.trust_penalty)
        self.penalized_trust_opinion = modified_trust
        return self.information_opinion.trust_discounting(modified_trust)

    def set_prior_probability(self, probability: float):
        self.trust_opinion.set_base_rate(probability)


class EBSL:
    "EBSL: Ensemble Binomial Subjective Logic"

    def __init__(self, conflict_threshold=0.05, max_penalty=0.5, b=1, trust_restore_speed=2, base_rate_choice: Literal["prior", "trust"] = "prior", _debug=False) -> None:
        """
        Collection of BSL_SM models. Enables prediction aggregation using subjective logic

        Parameters
        ----------
        conflict_threshold: The minimum value a model's conflict should have above the average conflict to reduce its trust

        max_penalty: The maximun penalty added to the model's trust opinion (disbelief)

        b: Inverse of speed of losing trust

        trust_restore_speed: Knowing that conflict counter increments with each conflict. This indicates how much it should decrement on lack of conflict

        base_rate_choice: How to choose the base rate. "prior" uses the last aggregated prediction probability.
        "trust" uses the probability produced by the currently most trusted model

        _debug: Enables debugging output
        """
        self.slmodels: list[BSL_SM] = []
        self.reference_opinion = Opinion()
        self.conflict_threshold = conflict_threshold
        self.max_penalty = max_penalty
        self.b = b
        self.trust_restore_speed = trust_restore_speed
        self._debug = _debug
        # Used in debug output iteration indicator
        self._debug_iteration_index = 0

        if base_rate_choice == "prior":
            self.base_rate_choice = 0
        elif base_rate_choice == "trust":
            self.base_rate_choice = 1
        else:
            print("Warning: Invalid base rate choice '%s' in constructor. Using default: \"prior\"..." %
                  (base_rate_choice))
            self.base_rate_choice = 0

        np.set_printoptions(legacy='1.25', precision=7, suppress=True)

    def add_model(self, model: BSL_SM):
        self.slmodels.append(model)

    def set_all_base_rates(self, base_rate):
        for slmodel in self.slmodels:
            slmodel.information_opinion.set_base_rate(base_rate)

    def get_penalty(self, nb_conflict):
        return self.max_penalty*nb_conflict/(nb_conflict + self.b)

    def get_all_opinions_and_ref(self, row):
        "Gets all original model opinions (by inference) then calculates the reference opinion (penalized)"
        discounted_opinions: list[Opinion] = []
        for slmodel in self.slmodels:
            # Get the information opinion
            slmodel.get_information_opinion(row)

        # Case where the base rate strategy was: highest trust
        # Find model with the current highest trust and use its belief as the base rate
        if self.base_rate_choice == 1:
            all_trust = [i.penalized_trust_opinion for i in self.slmodels]
            highest_trust_index = find_max_belief(all_trust)
            self.set_all_base_rates(
                self.slmodels[highest_trust_index].information_opinion._b)

        # The loop must have been splitted to avoid having to set the base rate twice (original, discounted)
        for slmodel in self.slmodels:
            # Calculate the information opinion after discounting with penalized trust
            discounted_opinion = slmodel.get_discounted_information_opinion()
            discounted_opinions.append(discounted_opinion)

        self.reference_opinion = average_fusion(discounted_opinions)

        if self._debug:
            print("Original information opinion")
            for i in range(len(self.slmodels)):
                print("Model %d: " % (i), end="")
                print(self.slmodels[i].information_opinion)

            print("\nDiscounted opinions")
            for i in range(len(self.slmodels)):
                print("Model %d: " % (i), end="")
                print(discounted_opinions[i])
            print("Reference opinion (penalized):",
                  self.reference_opinion, "\n")

    def get_all_conflicts(self) -> None:
        """Calculate conflict relative to the reference opinion. Results are stored in each model object"""
        for slmodel in self.slmodels:
            slmodel.conflict = slmodel.information_opinion.calculate_conflict(
                self.reference_opinion)

    def reevaluate_trust(self) -> None:
        """Updates the trust for each model according to the conflict"""
        all_conflict = [i.conflict for i in self.slmodels]
        average_conflict = np.average(all_conflict)
        distance_to_average_conf = np.subtract(all_conflict, average_conflict)

        for i in range(len(distance_to_average_conf)):
            if distance_to_average_conf[i] > self.conflict_threshold:
                self.slmodels[i].conflict_count += 1
                self.slmodels[i].trust_penalty = self.get_penalty(
                    self.slmodels[i].conflict_count)

            elif self.slmodels[i].conflict_count != 0:
                self.slmodels[i].conflict_count -= min(
                    self.trust_restore_speed, self.slmodels[i].conflict_count)
                self.slmodels[i].trust_penalty = self.get_penalty(
                    self.slmodels[i].conflict_count)

        if self._debug:
            print("Conflict:", all_conflict)
            print("Average conflict: %g" % average_conflict)
            print("Distance to average:", distance_to_average_conf)
            for i in range(len(self.slmodels)):
                cc = self.slmodels[i].conflict_count
                penalty = self.slmodels[i].trust_penalty
                print("Model %d: Conflict count = %g, penalty = %g" %
                      (i, cc, penalty))

    def get_final_prediction(self) -> float:
        "Calculates the final prediction using discounted information opinion. Updates the base rate for all model opinions"
        discounted_opinions: list[Opinion] = []
        for slmodel in self.slmodels:
            discounted_opinions.append(
                slmodel.get_discounted_information_opinion())

        final_opinion = average_fusion(discounted_opinions)
        # The base rate should be the same everywhere, so set it to the final opinion (or it will stay 0)
        final_opinion.set_base_rate(discounted_opinions[0]._a)

        if self._debug:
            print("\nFinal opinion:", final_opinion)
            # Projected probability is made of 2 parts: belief and contribution of prior probability
            print("Base rate contribution: %g" %
                  (final_opinion._a * final_opinion._u))

        prob = final_opinion.projected_probability()

        # Set the base rate for all models to be the newly calculated projected probability
        # According to the choice taken earlier
        if self.base_rate_choice == 0:
            self.set_all_base_rates(prob)

        return prob

    def run_once(self, row) -> float:
        if self._debug:
            print("--------------------------------------------")
            print("Iteration", self._debug_iteration_index)
            self._debug_iteration_index += 1

        # First run inference for each model, estimate opinions and their penalized average
        self.get_all_opinions_and_ref(row)
        # Find conflict between undiscounted opinions and penalized average
        self.get_all_conflicts()
        # Based on conflicts, find new trust values
        self.reevaluate_trust()
        return self.get_final_prediction()

    def predict_proba(self, X) -> np.ndarray:
        """
        Probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y_prob : ndarray of shape (n_samples, n_classes = 2)
            The predicted probability of the sample for each class, in order (class 0, class 1)
        """
        # I took that from sklearn documentation (with modifications)

        # Convert whatever was received to a numpy array
        # Should make lists, tuples, dataframes, arrays all usable
        X = np.array(X)
        nb_rows = len(X)
        y_prob = np.empty((nb_rows, 2))
        for input_row in range(nb_rows):
            class1 = self.run_once(X[input_row])
            y_prob[input_row][0] = 1.-class1
            y_prob[input_row][1] = class1

        return y_prob

    def predict(self, X) -> np.ndarray:
        """Predict using the ensemble of models added.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples)
        """
        # Convert whatever was received to a numpy array
        # Should make lists, tuples, dataframes, arrays all usable
        X = np.array(X)
        nb_rows = len(X)
        results = np.empty(nb_rows)
        for input_row in range(nb_rows):
            class1 = self.run_once(X[input_row])
            results[input_row] = round(class1)

        return results

    def get_raw_predictions(self, X):
        """
        Get the original predictions for all samples in X without running subjective logic algorithms

        Useful for testing against other multi-classifier models
        """
        X = np.array(X)
        nb_rows = len(X)

        all_results: list[np.ndarray] = []
        for slmodel in self.slmodels:
            all_results.append(slmodel.predict_proba(X))

        return np.concatenate(all_results, axis=1)
