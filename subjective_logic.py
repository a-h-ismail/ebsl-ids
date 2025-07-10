#!/usr/bin/env python3
from array import array
import copy
from typing import Literal
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef


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

    def __init__(self, model, scaler, trust_opinion=None, name="") -> None:
        """
        Creates the building blocks required for Ensemble Binomial Subjective Logic.
        It encapsulates an ML model, its scaler and manages subjective logic opinions (information, trust)

        Parameters
        ----------
        model: Any binary model providing methods predict() and predict_proba() like sklearn

        scaler: Should provide the transform() method like sklearn

        trust_opinion: Indicates the trustworthiness of a model. Affects the contribution of each model to the final prediction.

        name: The model name. If none is provided, a random one is generated
        """
        self.model = model
        self.scaler = scaler
        if trust_opinion is not None:
            self.trust_opinion = trust_opinion
        else:
            self.trust_opinion = Opinion()

        self.modified_trust = copy.copy(self.trust_opinion)
        self.trust_penalty = 0.
        # The opinion is for class 1
        # Set uncertainty to 0 here so we don't need to keep setting it to 0 everytime we get new information
        self.information_opinion = Opinion(1, 0, 0)
        self.discounted_information_opinion = Opinion()
        self.conflict = 0.
        self.conflict_count = 0.
        self.pcumulative_conflict = 0
        self.pconflict_TP = 0
        self.ncumulative_conflict = 0
        self.nconflict_TN = 0
        self.nclass_bonus = 0.
        self.pclass_bonus = 0.
        self.curr_bonus = 0.
        # The prediction cache is maintained here, but the current index is in EBSL
        self.prediction_cache: array
        # Generate a random unique name if the user doesn't provide a name
        if name == "":
            self.name = str(uuid4()).replace('-', '')[:16]
        else:
            self.name = name

    def trust_from_mcc(self, mcc: float, w=2):
        """Sets the trust opinion of this model using its Matthews correlation coefficient (MCC)"""
        scale = 100/(100+w)
        self.trust_opinion.set_parameters(mcc*scale, (1-mcc)*scale, 1-scale)

    def set_bonuses(self, nclass_bonus: float = 0, pclass_bonus: float = 0):
        self.pclass_bonus = pclass_bonus
        self.nclass_bonus = nclass_bonus

    def predict_proba_to_cache(self, samples):
        """
        Calls predict_proba of the underlying model after scaling the input samples
        """
        if self.scaler is not None:
            samples = samples[self.scaler.get_feature_names_out()]
            samples = self.scaler.transform(samples)
        # Convert to a tuple for the most efficient element by element access in Python
        self.prediction_cache = array('f', self.model.predict_proba(samples)[:, 1])
        self.ncumulative_conflict = self.pcumulative_conflict = 0

    def get_prediction(self, index: int) -> float:
        return self.prediction_cache[index]

    def get_information_opinion(self, index):
        """Generates the belief opinion of this model based on the prediction probability"""
        p = self.get_prediction(index)
        self.information_opinion._b = p
        self.information_opinion._d = 1-p

    def get_discounted_information_opinion(self) -> Opinion:
        """Calculates the discounted opinion according to the trust opinion modified with trust penalty and bonus.
        This function will update the internal state of the class too

        Warning: This function assumes you already called get_information_opinion() and updated the modified trust if necessary

        Returns: A new opinion after trust discounting (original opinion unchanged)
        """
        self.information_opinion.trust_discounting(self.modified_trust, out=self.discounted_information_opinion)
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

        id_col: Name of the column containing the "user" identifier. Necessary to track trust for each user separately

        _debug: Enables debugging output
        """
        self.slmodels: list[BSL_SM] = []
        self._slmodels_dict = {}
        self._reference_opinion = Opinion()
        self.conflict_threshold = conflict_threshold
        self.max_penalty = max_penalty
        self.b = b
        self.trust_restore_speed = trust_restore_speed
        self._debug = _debug
        # Used in debug output iteration indicator and for predictions in bulk
        self._cache_i = 0
        self._cache_max = 0
        # User ID in the previous iteration
        self._last_id = 0
        self.base_rate_choice_str = base_rate_choice
        self._last_final_opinion = Opinion()
        self._last_predict_proba = 0.
        self._empty_opinion = Opinion(0.1, 0.9, 0)
        # Store true labels to help with tuning
        self._true_labels: array
        # The debug switch that enables true labels comparison (so we can know if the model at conflict is right or not)
        self._compare_to_true_label = False
        # Track the classifier state per user
        self._state_store = dict()
        self._id_col = id_col

        if id_col == "":
            self._multi_user = False
        else:
            self._multi_user = True

        if base_rate_choice == "prior":
            self._base_rate_choice = 0
        elif base_rate_choice == "trust":
            self._base_rate_choice = 1
        else:
            print("Warning: Invalid base rate choice '%s' in constructor. Using default: \"prior\"..." % (base_rate_choice))
            self._base_rate_choice = 0

        np.set_printoptions(legacy='1.25', precision=7, suppress=True)

    def __str__(self) -> str:
        return "EBSL classifier: conflict_threshold=%g, max_penalty=%g, b=%g, trust_restore_speed=%g, base_rate_choice:\"%s\", nb_of_classifiers = %d" % (self.conflict_threshold, self.max_penalty, self.b, self.trust_restore_speed, self.base_rate_choice_str, len(self.slmodels))

    def add_model(self, model: BSL_SM):
        if model.name in self._slmodels_dict:
            raise ValueError("The Model with name %s already exists!" % model.name)
        self.slmodels.append(model)
        self._slmodels_dict[model.name] = model
        # Clear already stored states just in case
        self._state_store.clear()

    def trust_from_dataset_mcc(self, samples: pd.DataFrame, true_labels) -> None:
        """Set the trust of all Models in the ensemble from the MCC metrics of the provided dataset
        Note: This function fills the prediction cache for all models."""
        self._gen_prediction_cache(samples)
        true_labels = np.asarray(true_labels)
        for model in self.slmodels:
            mcc = matthews_corrcoef(true_labels, np.round(model.prediction_cache))
            model.trust_from_mcc(mcc)

    def get_model_by_name(self, name: str) -> BSL_SM:
        return self._slmodels_dict[name]

    def _set_all_base_rates(self, base_rate):
        """Set the base rate for all information opinions"""
        for slmodel in self.slmodels:
            slmodel.information_opinion.set_base_rate(base_rate)

    def _save_state(self, user_id):
        if user_id in self._state_store:
            # Update the stored state instead of allocating a new one with updated information
            state = self._state_store[user_id]
            state[0] = self._last_predict_proba
            conflict_counters = state[1]
            bonuses = state[2]
            # Store updated bonuses and conflict counters
            for i in range(len(conflict_counters)):
                conflict_counters[i] = self.slmodels[i].conflict_count
                bonuses[i] = self.slmodels[i].curr_bonus

        else:
            # Collect current modified trust opinions and store them separately
            conflict_counters = []
            bonuses = []
            for slmodel in self.slmodels:
                conflict_counters.append(slmodel.conflict_count)
                bonuses.append(slmodel.curr_bonus)

            self._state_store[user_id] = [self._last_predict_proba, conflict_counters, bonuses]

    def _load_state(self, user_id):
        if user_id in self._state_store:
            state = self._state_store[user_id]
            self._last_predict_proba = state[0]
            conflict_counters = state[1]
            bonuses = state[2]
            # Now set the base rate for all models information opinions (overwriting the old base rate)
            self._set_all_base_rates(self._last_predict_proba)
            # Restore the conflict counter and corresponding modified trust
            for i in range(len(conflict_counters)):
                slm = self.slmodels[i]
                slm.curr_bonus = bonuses[i]
                slm.conflict = self._get_penalty(conflict_counters[i])-bonuses[i]
                slm.trust_opinion.modify_trust(slm.conflict, out=slm.modified_trust)
        else:
            # Use defaults
            for slm in self.slmodels:
                self._last_predict_proba = 0.49
                self._set_all_base_rates(self._last_predict_proba)
                slm.conflict = slm.conflict_count = slm.curr_bonus = 0
                slm.modified_trust.copy(slm.trust_opinion)

    def _get_penalty(self, nb_conflict):
        return self.max_penalty*nb_conflict/(nb_conflict + self.b)

    def _get_all_opinions_and_ref(self):
        "Gets all original model opinions (by inference) then calculates the reference opinion if necessary"
        discounted_opinions: list[Opinion] = []
        for slmodel in self.slmodels:
            # Get the information opinion
            slmodel.get_information_opinion(self._cache_i)

        # Case where the base rate strategy was: highest trust
        # Find model with the current highest trust and use its belief as the base rate
        if self._base_rate_choice == 1:
            all_trust = [i.modified_trust for i in self.slmodels]
            highest_trust_index = find_max_belief(all_trust)
            self._set_all_base_rates(self.slmodels[highest_trust_index].information_opinion._b)

        for slmodel in self.slmodels:
            # Calculate the information opinion after discounting with modified trust
            discounted_opinion = slmodel.get_discounted_information_opinion()
            discounted_opinions.append(discounted_opinion)

        # Perform average fusion only in case of base rate source = trust
        if self._base_rate_choice == 1:
            average_fusion(discounted_opinions, out=self._reference_opinion)

        if self._debug:
            print("* Initialization")
            print("Information opinions:")
            for slmodel in self.slmodels:
                print("Model %s:" % slmodel.name, slmodel.information_opinion)

            print("\n* Before trust update")
            print("Discounted information opinions:")
            for slmodel in self.slmodels:
                print("Model %s:" % slmodel.name, slmodel.discounted_information_opinion)
            print("Reference opinion:", self._reference_opinion, "\n")

    def _get_all_conflicts(self) -> None:
        """Calculate conflict relative to the reference opinion. Results are stored in each model object"""
        if self._base_rate_choice == 0:
            # Remember that the base rate is the last prior probability
            a = self.slmodels[0].information_opinion._a
            self._reference_opinion._b = a
            self._reference_opinion._d = 1 - a

        # If base rate choice is 1, the ref opinion is stored earlier in self
        # Otherwise (prior mode) it is updated as you see in the above lines
        reference = self._reference_opinion
        if self._base_rate_choice == 1:
            p = self._last_predict_proba
        else:
            p = reference.projected_probability()
        u = reference._u

        for slmodel in self.slmodels:
            slmodel.conflict = slmodel.information_opinion.calculate_conflict_fast(p, u)

    def _reevaluate_trust(self) -> None:
        """Updates the trust for each model according to the conflict"""
        all_conflict = [i.conflict for i in self.slmodels]
        average_conflict = sum(all_conflict)/len(all_conflict)
        distance_to_average_conf = [i - average_conflict for i in all_conflict]

        for i in range(len(distance_to_average_conf)):
            slmodel = self.slmodels[i]
            if distance_to_average_conf[i] > self.conflict_threshold:
                slmodel.conflict_count += 1
                slmodel.trust_penalty = self._get_penalty(slmodel.conflict_count)
                # The model is predicting positive class, so use the pclass bonus
                if slmodel.information_opinion._b >= 0.5:
                    slmodel.pcumulative_conflict += 1
                    # Used for bonus tuning (to know which models are worth giving bonuses)
                    if self._compare_to_true_label and self._true_labels[self._cache_i] == 1:
                        slmodel.pconflict_TP += 1

                    slmodel.trust_penalty -= slmodel.pclass_bonus
                    slmodel.curr_bonus = slmodel.pclass_bonus
                else:
                    slmodel.ncumulative_conflict += 1
                    # Also for bonus tuning
                    if self._compare_to_true_label and self._true_labels[self._cache_i] == 0:
                        slmodel.nconflict_TN += 1

                    slmodel.trust_penalty -= slmodel.nclass_bonus
                    slmodel.curr_bonus = slmodel.nclass_bonus

                # Recalculate the modified trust opinion immediately (to avoid always recalculating all trust opinions later)
                slmodel.trust_opinion.modify_trust(slmodel.trust_penalty, out=slmodel.modified_trust)
                # And the new discounted information opinion
                slmodel.get_discounted_information_opinion()

            elif slmodel.conflict_count != 0:
                slmodel.conflict_count -= min(self.trust_restore_speed, slmodel.conflict_count)

                if slmodel.conflict_count == 0:
                    # Reset both counters
                    slmodel.trust_penalty = slmodel.curr_bonus = 0
                else:
                    slmodel.trust_penalty = self._get_penalty(slmodel.conflict_count) - slmodel.curr_bonus

                # Same optimization as above
                slmodel.trust_opinion.modify_trust(slmodel.trust_penalty, out=slmodel.modified_trust)
                slmodel.get_discounted_information_opinion()

        if self._debug:
            print("* Conflict statistics")
            print("Conflict:", ["{0:0.3f}".format(i) for i in all_conflict])
            print("Average conflict = %.3g" % average_conflict)
            print("Distance to average:", ["{0:0.3f}".format(i) for i in distance_to_average_conf])
            for model in self.slmodels:
                name = model.name
                cc = model.conflict_count
                penalty = model.trust_penalty
                bonus = model.curr_bonus
                print("Model %s: Conflict count = %g, curr_penalty = %.3g, bonus = %.3g" % (name, cc, penalty, bonus))

            print("\n* After trust update")
            print("Discounted information opinions:")
            for slmodel in self.slmodels:
                print("Model %s:" % slmodel.name, slmodel.discounted_information_opinion)

    def _get_final_prediction(self) -> float:
        "Calculates the final prediction using discounted information opinion. Updates the base rate for all model opinions"
        discounted_opinions: list[Opinion] = []
        for slmodel in self.slmodels:
            discounted_opinions.append(slmodel.discounted_information_opinion)

        final_opinion = self._last_final_opinion
        average_fusion(discounted_opinions, out=final_opinion)
        # The base rate should be the same everywhere, so set it to the final opinion (or it will stay 0)
        final_opinion.set_base_rate(discounted_opinions[0]._a)

        prob = final_opinion.projected_probability()
        self._last_predict_proba = prob

        if self._debug:
            print("\n* Final opinion:", final_opinion)
            # Projected probability is made of 2 parts: belief and contribution of prior probability
            print("Base rate contribution = %.3g" % (final_opinion._a * final_opinion._u))
            print("Class 1 Probability = %.3g" % prob)

        # UserID awareness when needed
        if self._multi_user:
            self._last_id = self._id_list[self._cache_i]

        return prob

    def _gen_prediction_cache(self, samples: pd.DataFrame):
        """Fills the prediction cache of all models for the current samples"""
        # When in multi user mode, we track the trust penalties and previous final opinion per user
        if self._multi_user:
            self._id_list = array('Q', samples[self._id_col].array)

        for model in self.slmodels:
            model.predict_proba_to_cache(samples)
        self._cache_i = 0
        self._cache_max = samples.shape[0]

    def _run_once(self) -> float:
        if self._debug:
            print("--------------------------------------------")
            print("Iteration", self._cache_i)

        if self._multi_user:
            current_id = self._id_list[self._cache_i]
            if current_id == self._last_id and self._base_rate_choice == 0:
                # The user ID didn't change from the last iteration
                # And our base rate source is the last prediction probability
                # We only need to update the base rate in this case
                self._set_all_base_rates(self._last_predict_proba)
            else:
                self._save_state(self._last_id)
                self._load_state(current_id)
        else:
            self._set_all_base_rates(self._last_predict_proba)

        # Estimate opinions and their modified average (assuming all inference is done earlier)
        self._get_all_opinions_and_ref()
        # Find conflict between undiscounted opinions and modified average
        self._get_all_conflicts()
        # Based on conflicts, find new trust values
        self._reevaluate_trust()
        return self._get_final_prediction()

    def auto_tune(self, samples, true_labels, bonus_step=0.2, _show_progress=False):
        """Finds the best model trust bonuses for the given dataset. Sets models trust opinion from their MCC."""
        self.trust_from_dataset_mcc(samples, true_labels)
        # Reset bonus for all models and set initial trust from MCC
        for slmodel in self.slmodels:
            slmodel.set_bonuses(0, 0)

        # Sort models in the internal list according to the trust
        self.slmodels.sort(key=lambda x: x.trust_opinion._b, reverse=True)

        # Perform a run without bonuses to get a baseline of models behavior under conflict
        predicted = self.predict(samples, True, true_labels)
        old_mcc = matthews_corrcoef(true_labels, predicted)

        if _show_progress:
            print("Baseline MCC (no bonuses): %g" % old_mcc)

        # Traversing models in descending order
        for model in self.slmodels:
            p_bonus = n_bonus = old_bonus = 0

            # Loop to find the best positive class bonus
            if model.pcumulative_conflict > 0:
                dist = model.pconflict_TP/model.pcumulative_conflict - 0.5
                while p_bonus < 1 and p_bonus > -1:
                    old_bonus = p_bonus
                    if dist > 0:
                        p_bonus = min(1, p_bonus+bonus_step)
                    else:
                        p_bonus = max(-1, p_bonus-bonus_step)
                    model.set_bonuses(0, p_bonus)
                    predicted = self.predict(samples, True, true_labels)
                    new_mcc = matthews_corrcoef(true_labels, predicted)
                    # If our increment/decrement didn't provide improvements, roll it back
                    if new_mcc < old_mcc:
                        p_bonus = old_bonus
                        model.set_bonuses(0, old_bonus)
                        break
                    old_mcc = new_mcc

                if _show_progress:
                    print("Model %s received pclass bonus = %g, MCC = %g" % (model.name, model.pclass_bonus, old_mcc))

            # The same algorithm but for the negative class bonus
            if model.ncumulative_conflict > 0:
                dist = model.nconflict_TN/model.ncumulative_conflict - 0.5

                while n_bonus < 1 and n_bonus > -1:
                    old_bonus = n_bonus
                    if dist > 0:
                        n_bonus = min(1, n_bonus+bonus_step)
                    else:
                        n_bonus = max(-1, n_bonus-bonus_step)
                    model.set_bonuses(n_bonus, p_bonus)
                    predicted = self.predict(samples, True, true_labels)
                    new_mcc = matthews_corrcoef(true_labels, predicted)
                    # If our increment/decrement didn't provide improvements, roll it back
                    if new_mcc < old_mcc:
                        n_bonus = old_bonus
                        model.set_bonuses(old_bonus, p_bonus)
                        break
                    old_mcc = new_mcc

                if _show_progress:
                    print("Model %s received nclass bonus = %g, MCC = %g" % (model.name, model.nclass_bonus, old_mcc))

    def predict(self, X, _keep_caches=False, _true_labels=None) -> np.ndarray:
        """Predict using the ensemble of models added.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples)
        """
        if not _keep_caches or self._cache_i == 0:
            self._gen_prediction_cache(X)

        if _true_labels is not None:
            self._true_labels = _true_labels
            self._compare_to_true_label = True
        else:
            self._compare_to_true_label = False

        self._cache_i = 0
        self._state_store.clear()
        # Clear old conflict statistics
        for m in self.slmodels:
            m.pcumulative_conflict = m.pconflict_TP = 0
            m.ncumulative_conflict = m.nconflict_TN = 0

        # Set once here and forget it (in case of prior not trust)
        if self._base_rate_choice == 0:
            self._reference_opinion._u = 0
        nb_rows = X.shape[0]
        results = array('f', np.empty(nb_rows, dtype=float))
        for input_row in range(nb_rows):
            self._cache_i = input_row
            class1 = self._run_once()
            results[input_row] = round(class1)

        return np.asarray(results)

    def _merge_caches(self):
        """Combines all predictions"""
        predictions: list[array] = []
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
