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
            offset = -self._d

        if out is not None:
            out._b = self._b - offset
            out._d = self._d + offset
            return out

        if inplace:
            self._b -= offset
            self._d += offset
            return self
        else:
            penalized_trust = copy.copy(self)
            if offset == 0:
                return penalized_trust
            else:
                penalized_trust._b -= offset
                penalized_trust._d += offset
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

    def copy(self, source):
        """Copies source into this object inplace"""
        self._b = source._b
        self._d = source._d
        self._u = source._u
        self._a = source._a

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
        self.discounted_information_opinion = Opinion()
        self.conflict = 0.
        self.conflict_count = 0.
        self.cumulative_conflict = 0
        # The prediction cache is maintained here, but the current index is in EBSL
        self.prediction_cache = ()

    def predict_proba_to_cache(self, samples):
        """
        Calls predict_proba of the underlying model after scaling the input samples
        """
        if self.scaler is not None:
            samples = samples[self.scaler.get_feature_names_out()]
            samples = self.scaler.transform(samples)
        # Convert to a tuple for the most efficient element by element access in Python
        self.prediction_cache = tuple(self.model.predict_proba(samples)[:, 1])

    def get_prediction(self, index: int) -> float:
        return self.prediction_cache[index]

    def get_information_opinion(self, index):
        """Generates the belief opinion of this model based on the prediction probability"""
        p = self.get_prediction(index)
        self.information_opinion.set_parameters(p, 1-p, 0)

    def get_discounted_information_opinion(self) -> Opinion:
        """Calculates the discounted opinion according to the trust opinion modified with trust penalty

        Warning: This function assumes you already called get_information_opinion() and updated the penalized trust if necessary

        Returns: A new opinion after trust discounting (original opinion unchanged)
        """
        self.information_opinion.trust_discounting(
            self.penalized_trust_opinion, out=self.discounted_information_opinion)
        return self.discounted_information_opinion

    def set_prior_probability(self, probability: float):
        self.trust_opinion.set_base_rate(probability)


class EBSL:
    "EBSL: Ensemble Binomial Subjective Logic"

    def __init__(self, conflict_threshold=0.15, max_penalty=0.5, b=1., trust_restore_speed=0.5, base_rate_choice: Literal["prior", "trust"] = "prior", id_col: str = "", _debug=False) -> None:
        """
        Collection of BSL_SM models. Enables prediction aggregation using subjective logic

        Parameters
        ----------
        conflict_threshold: The minimum value a model's conflict should have above the average conflict to reduce its trust

        max_penalty: The maximun penalty added to the model's trust opinion (disbelief)

        b: Inverse of speed of losing trust

        trust_restore_speed: Indicates trust restoration step size on lack of conflict (mistrust step size is 1)

        base_rate_choice: How to choose the base rate. "prior" uses the last aggregated prediction probability.
        "trust" uses the probability produced by the currently most trusted model

        id_col: Name of the column containing the "user" identifier, required to use "prior" mode

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
        self.cache_i = 0
        self.cache_max = 0
        self.last_id = 0
        self._base_rate_choice_str = base_rate_choice
        self.last_final_opinion = Opinion()
        self._empty_opinion = Opinion()

        if base_rate_choice == "prior":
            self.base_rate_choice = 0
            # Tracking the base rate per user ID is necessary for the correct operation of "prior" mode
            self.state_store = dict()
            self.id_col = id_col
        elif base_rate_choice == "trust":
            self.base_rate_choice = 1
        else:
            print("Warning: Invalid base rate choice '%s' in constructor. Using default: \"prior\"..." % (base_rate_choice))
            self.base_rate_choice = 0

        np.set_printoptions(legacy='1.25', precision=7, suppress=True)

    def __str__(self) -> str:
        return "EBSL classifier: conflict_threshold=%g, max_penalty=%g, b=%g, trust_restore_speed=%g, base_rate_choice:\"%s\", nb_of_classifiers = %d" % (self.conflict_threshold, self.max_penalty, self.b, self.trust_restore_speed, self._base_rate_choice_str, len(self.slmodels))

    def add_model(self, model: BSL_SM):
        self.slmodels.append(model)

    def set_all_base_rates(self, base_rate):
        for slmodel in self.slmodels:
            slmodel.information_opinion.set_base_rate(base_rate)

    def _save_state(self, user_id):
        if user_id in self.state_store:
            # Update the stored state instead of allocating a new one with updated information
            final_opinion, conflict_counters = self.state_store[user_id]
            for i in range(len(conflict_counters)):
                conflict_counters[i] = self.slmodels[i].conflict_count
            final_opinion.copy(self.last_final_opinion)
        else:
            # Collect current penalized trust opinions and store them separately
            conflict_counters = []
            for slmodel in self.slmodels:
                conflict_counters.append(slmodel.conflict_count)

            self.state_store[user_id] = (copy.copy(self.last_final_opinion), conflict_counters)

    def _load_state(self, user_id):
        if user_id in self.state_store:
            (final_opinion, conflict_counters) = self.state_store[user_id]
            self.last_final_opinion.copy(final_opinion)
            # Now set the base rate for all models information opinions (overwriting the old base rate)
            self.set_all_base_rates(round(self.last_final_opinion.projected_probability()))
            # Restore the conflict counter and corresponding penalized trust
            for i in range(len(conflict_counters)):
                slm = self.slmodels[i]
                slm.conflict = self.get_penalty(conflict_counters[i])
                slm.trust_opinion.modify_trust(slm.conflict, out=slm.penalized_trust_opinion)
        else:
            # Use defaults
            for slm in self.slmodels:
                slm.conflict = slm.conflict_count = 0
                slm.penalized_trust_opinion.copy(slm.trust_opinion)
                self.set_all_base_rates(0.49)
                self.last_final_opinion = self._empty_opinion

    def get_penalty(self, nb_conflict):
        return self.max_penalty*nb_conflict/(nb_conflict + self.b)

    def get_all_opinions_and_ref(self):
        "Gets all original model opinions (by inference) then calculates the reference opinion (penalized)"
        discounted_opinions: list[Opinion] = []
        for slmodel in self.slmodels:
            # Get the information opinion
            slmodel.get_information_opinion(self.cache_i)

        # Case where the base rate strategy was: highest trust
        # Find model with the current highest trust and use its belief as the base rate
        if self.base_rate_choice == 1:
            all_trust = [i.penalized_trust_opinion for i in self.slmodels]
            highest_trust_index = find_max_belief(all_trust)
            self.set_all_base_rates(self.slmodels[highest_trust_index].information_opinion._b)

        for slmodel in self.slmodels:
            # Calculate the information opinion after discounting with penalized trust
            discounted_opinion = slmodel.get_discounted_information_opinion()
            discounted_opinions.append(discounted_opinion)

        if self.base_rate_choice == 1:
            average_fusion(discounted_opinions, out=self.reference_opinion)

        if self._debug:
            print("Original information opinion")
            for i in range(len(self.slmodels)):
                print("Model %d: " % (i), end="")
                print(self.slmodels[i].information_opinion)

            print("\nDiscounted opinions (before update)")
            for i in range(len(self.slmodels)):
                print("Model %d: " % (i), end="")
                print(discounted_opinions[i])
            print("Reference opinion (penalized before update):",
                  self.reference_opinion, "\n")

    def get_all_conflicts(self) -> None:
        """Calculate conflict relative to the reference opinion. Results are stored in each model object"""
        if self.base_rate_choice == 0:
            a = self.slmodels[0].information_opinion._a
            self.reference_opinion._b = a
            self.reference_opinion._d = 1 - a
            self.reference_opinion._u = 0

        reference = self.reference_opinion

        for slmodel in self.slmodels:
            slmodel.conflict = slmodel.information_opinion.calculate_conflict(reference)

    def reevaluate_trust(self) -> None:
        """Updates the trust for each model according to the conflict"""
        all_conflict = [i.conflict for i in self.slmodels]
        average_conflict = np.average(all_conflict)
        distance_to_average_conf = np.subtract(all_conflict, average_conflict)

        for i in range(len(distance_to_average_conf)):
            slmodel = self.slmodels[i]
            if distance_to_average_conf[i] > self.conflict_threshold:
                slmodel.conflict_count += 1
                slmodel.cumulative_conflict += 1
                slmodel.trust_penalty = self.get_penalty(slmodel.conflict_count)

                # Recalculate the penalized trust opinion immediately (to avoid always recalculating all opinions)
                slmodel.trust_opinion.modify_trust(slmodel.trust_penalty, out=slmodel.penalized_trust_opinion)
                # And the new discounted information opinion
                slmodel.get_discounted_information_opinion()

            elif slmodel.conflict_count != 0:
                slmodel.conflict_count -= min(self.trust_restore_speed, slmodel.conflict_count)
                slmodel.trust_penalty = self.get_penalty(slmodel.conflict_count)

                slmodel.trust_opinion.modify_trust(slmodel.trust_penalty, out=slmodel.penalized_trust_opinion)
                slmodel.get_discounted_information_opinion()

        if self._debug:
            print("Conflict:", ["{0:0.7f}".format(i) for i in all_conflict])
            print("Average conflict: %g" % average_conflict)
            print("Distance to average:", distance_to_average_conf)
            for i in range(len(self.slmodels)):
                cc = self.slmodels[i].conflict_count
                penalty = self.slmodels[i].trust_penalty
                print("Model %d: Conflict count = %g, penalty = %g" % (i, cc, penalty))

    def get_final_prediction(self) -> float:
        "Calculates the final prediction using discounted information opinion. Updates the base rate for all model opinions"
        discounted_opinions: list[Opinion] = []
        for slmodel in self.slmodels:
            discounted_opinions.append(slmodel.discounted_information_opinion)

        final_opinion = self.last_final_opinion
        average_fusion(discounted_opinions, out=final_opinion)
        # The base rate should be the same everywhere, so set it to the final opinion (or it will stay 0)
        final_opinion.set_base_rate(discounted_opinions[0]._a)

        prob = final_opinion.projected_probability()

        if self._debug:
            print("\nDiscounted opinions (after update)")
            for i in range(len(self.slmodels)):
                print("Model %d: " % (i), end="")
                print(discounted_opinions[i])
            print("Reference opinion (penalized after update):", self.reference_opinion)
            print("\nFinal opinion:", final_opinion)
            # Projected probability is made of 2 parts: belief and contribution of prior probability
            print("Base rate contribution: %g" % (final_opinion._a * final_opinion._u))
            print("Probability = %g" % prob)

        # Set the base rate for all models to be the newly calculated projected probability
        # According to the choice taken earlier
        if self.base_rate_choice == 0:
            self.last_id = self.id_list[self.cache_i]

        # Preparing for the next iteration
        self.cache_i += 1

        return prob

    def _gen_predict_cache(self, samples):
        """Fills the prediction cache of all models for the current samples"""
        self.id_list = samples[self.id_col]
        for model in self.slmodels:
            model.predict_proba_to_cache(samples)
        self.cache_i = 0
        self.cache_max = samples.shape[0]

    def run_once(self) -> float:
        if self._debug:
            print("--------------------------------------------")
            print("Iteration", self.cache_i)

        if self.base_rate_choice == 0:
            current_id = self.id_list[self.cache_i]
            if current_id == self.last_id:
                # We only need to update the base rate in this case
                self.set_all_base_rates(self.last_final_opinion.projected_probability())
            else:
                self._save_state(self.last_id)
                self._load_state(current_id)

        # Estimate opinions and their penalized average (assuming all inference is done earlier)
        self.get_all_opinions_and_ref()
        # Find conflict between undiscounted opinions and penalized average
        self.get_all_conflicts()
        # Based on conflicts, find new trust values
        self.reevaluate_trust()
        return self.get_final_prediction()

    def predict(self, X) -> np.ndarray:
        """Predict using the ensemble of models added.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples)
        """
        self._gen_predict_cache(X)
        nb_rows = X.shape[0]
        results = np.empty(nb_rows)
        for input_row in range(nb_rows):
            class1 = self.run_once()
            results[input_row] = round(class1)

        return results

    def _merge_caches(self):
        """Combines all predictions"""
        predictions: list[tuple] = []
        for model in self.slmodels:
            predictions.append(model.prediction_cache)
        return np.array(predictions).T

    def _hard_vote(self):
        caches = self._merge_caches()
        votes = np.round(caches)
        sum_votes = np.sum(votes, axis=1)
        threshold = votes.shape[1]/2
        pred = np.where(sum_votes > threshold, 1, 0)
        return pred

    def _soft_vote(self):
        caches = self._merge_caches()
        sum_votes = np.sum(caches, axis=1)
        threshold = caches.shape[1]/2
        pred = np.where(sum_votes > threshold, 1, 0)
        return pred
