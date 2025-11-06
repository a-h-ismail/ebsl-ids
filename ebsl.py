#!/usr/bin/env python3
from array import array
import copy
from typing import Literal
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from binomial_opinion import *
from binomial_opinion import _uncertainty_product


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
            self._trust_opinion = trust_opinion
        else:
            self._trust_opinion = Opinion()
        # The modified trust should be initialized to be equal to the initial trust
        self._modified_trust = copy.copy(self._trust_opinion)
        # Trust offset is penalty - bonus
        self._trust_offset = 0.
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

    def set_initial_trust_opinion(self, b, d, u):
        self._trust_opinion.set_parameters(b, d, u)
        self._modified_trust.set_parameters(b, d, u)

    def trust_from_mcc(self, mcc: float, w=2):
        """Sets the trust opinion of this model using its Matthews correlation coefficient (MCC)"""
        scale = 100/(100+w)
        self.set_initial_trust_opinion(mcc*scale, (1-mcc)*scale, 1-scale)

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
        self.information_opinion.trust_discounting(self._modified_trust, out=self.discounted_information_opinion)
        return self.discounted_information_opinion

    def set_prior_probability(self, probability: float):
        self._trust_opinion.set_base_rate(probability)


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

        id_col: Name of the column containing the flow identifier. Necessary to track trust for each flow separately

        _debug: Enables debugging output
        """
        self.slmodels: list[BSL_SM] = []
        self._slmodels_dict = {}
        self._reference_opinion = Opinion()
        self.conflict_threshold = conflict_threshold
        self.max_penalty = max_penalty
        self.b = b
        self.trust_restore_speed = trust_restore_speed
        self._base_weights: array
        self._debug = _debug
        # Used in debug output iteration indicator and for predictions in bulk
        self._cache_i = 0
        self._cache_max = 0
        # Flow ID in the previous iteration
        self._last_id = 0
        self.base_rate_choice_str = base_rate_choice
        self._last_final_opinion = Opinion()
        self._last_predict_proba = 0.
        self._empty_opinion = Opinion(0.1, 0.9, 0)
        # Store true labels to help with tuning
        self._true_labels: array
        # The debug switch that enables true labels comparison (so we can know if the model at conflict is right or not)
        self._compare_to_true_label = False
        # Track the classifier state per flow
        self._state_store = dict()
        self._id_col = id_col

        if id_col == "":
            self._multi_flow = False
        else:
            self._multi_flow = True

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

    def init_base_weights(self):
        model_count = len(self.slmodels)

        self._base_weights = array('f')
        # Find the common denominator
        denominator = 0
        for i in range(model_count):
            tmp = 1
            for j in range(model_count):
                if j == i:
                    continue
                else:
                    tmp *= 1-self.slmodels[j]._trust_opinion._b
            denominator += tmp

        for i in range(model_count):
            numerator = self.slmodels[i]._trust_opinion._b
            for j in range(model_count):
                if i == j:
                    continue
                else:
                    numerator *= (1-self.slmodels[j]._trust_opinion._b)
            self._base_weights.append(numerator/denominator)

        # Take the weight of uncertainty into account
        self._base_weights.append(1-sum(self._base_weights))

    def _save_state(self, flow_id):
        if flow_id in self._state_store:
            # Update the stored state instead of allocating a new one with updated information
            state = self._state_store[flow_id]
            state[0] = self._last_predict_proba
            conflict_counters = state[1]
            bonuses = state[2]
            trust_offsets = state[3]
            models = self.slmodels
            # Store updated bonuses and conflict counters
            for i in range(len(conflict_counters)):
                conflict_counters[i] = models[i].conflict_count
                bonuses[i] = models[i].curr_bonus
                trust_offsets[i] = models[i]._trust_offset

        else:
            # Collect current modified trust opinions and store them separately
            conflict_counters = array('f')
            bonuses = array('f')
            trust_offsets = array('f')
            for slmodel in self.slmodels:
                conflict_counters.append(slmodel.conflict_count)
                bonuses.append(slmodel.curr_bonus)
                trust_offsets.append(slmodel._trust_offset)

            self._state_store[flow_id] = [self._last_predict_proba, conflict_counters, bonuses, trust_offsets]

    def _load_state(self, flow_id):
        if flow_id in self._state_store:
            state = self._state_store[flow_id]
            self._last_predict_proba = state[0]
            conflict_counters = state[1]
            bonuses = state[2]
            trust_offsets = state[3]
            # Now set the base rate for all models information opinions (overwriting the old base rate)
            self._set_all_base_rates(self._last_predict_proba)
            # Restore the conflict counter and corresponding modified trust
            for i in range(len(self.slmodels)):
                slm = self.slmodels[i]
                slm.curr_bonus = bonuses[i]
                slm.conflict_count = conflict_counters[i]
                slm._trust_offset = trust_offsets[i]
                slm._trust_opinion.modify_trust(slm._trust_offset, out=slm._modified_trust)
        else:
            # Use defaults
            for slm in self.slmodels:
                self._last_predict_proba = 0.49
                self._set_all_base_rates(self._last_predict_proba)
                slm.conflict = slm.conflict_count = slm.curr_bonus = slm._trust_offset = 0
                slm._modified_trust.copy(slm._trust_opinion)

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
            all_trust = [i._modified_trust for i in self.slmodels]
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
                # Applying penalty
                slmodel.conflict_count += 1
                slmodel._trust_offset = self._get_penalty(slmodel.conflict_count)
                # The model is predicting positive class, so use the pclass bonus
                if slmodel.information_opinion._b >= 0.5:
                    slmodel.pcumulative_conflict += 1
                    # Used for bonus tuning (to know which models are worth giving bonuses)
                    if self._compare_to_true_label and self._true_labels[self._cache_i] == 1:
                        slmodel.pconflict_TP += 1

                    slmodel._trust_offset -= slmodel.pclass_bonus
                    slmodel.curr_bonus = slmodel.pclass_bonus
                else:
                    slmodel.ncumulative_conflict += 1
                    # Also for bonus tuning
                    if self._compare_to_true_label and self._true_labels[self._cache_i] == 0:
                        slmodel.nconflict_TN += 1

                    slmodel._trust_offset -= slmodel.nclass_bonus
                    slmodel.curr_bonus = slmodel.nclass_bonus

                # Recalculate the modified trust opinion immediately (to avoid always recalculating all trust opinions later)
                slmodel._trust_opinion.modify_trust(slmodel._trust_offset, out=slmodel._modified_trust)
                # And the new discounted information opinion
                slmodel.get_discounted_information_opinion()

            elif slmodel.conflict_count != 0:
                # Gradually restoring trust
                slmodel.conflict_count -= min(self.trust_restore_speed, slmodel.conflict_count)

                if slmodel.conflict_count == 0:
                    # Reset both counters
                    slmodel._trust_offset = slmodel.curr_bonus = 0
                else:
                    slmodel._trust_offset = self._get_penalty(slmodel.conflict_count) - slmodel.curr_bonus

                # Same optimization as above
                slmodel._trust_opinion.modify_trust(slmodel._trust_offset, out=slmodel._modified_trust)
                slmodel.get_discounted_information_opinion()

        if self._debug:
            print("* Conflict statistics")
            print("Conflict:", ["{0:0.3f}".format(i) for i in all_conflict])
            print("Average conflict = %.3g" % average_conflict)
            print("Distance to average:", ["{0:0.3f}".format(i) for i in distance_to_average_conf])
            for model in self.slmodels:
                name = model.name
                cc = model.conflict_count
                penalty = model._trust_offset
                bonus = model.curr_bonus
                print("Model %s: Conflict count = %g, curr_penalty = %.3g, bonus = %.3g" % (name, cc, penalty, bonus))

            print("\n* After trust update")
            print("Discounted information opinions:")
            for slmodel in self.slmodels:
                print("Model %s:" % slmodel.name, slmodel.discounted_information_opinion)

            # Store dist to average conflict, penalties and uncertainty
            for i in range(len(self.slmodels)):
                self._slm_dist_to_avg[i].append(distance_to_average_conf[i])
                self._slm_penalties[i].append(self.slmodels[i]._trust_offset)
                self._slm_uncertainty[i].append(self.slmodels[i].discounted_information_opinion._u)

    def _get_final_prediction(self) -> float:
        "Calculates the final prediction using discounted information opinion. Updates the base rate for all model opinions"
        discounted_opinions: list[Opinion] = []
        for slmodel in self.slmodels:
            discounted_opinions.append(slmodel.discounted_information_opinion)

        final_opinion = self._last_final_opinion
        average_fusion(discounted_opinions, out=final_opinion)
        # The base rate should be the same everywhere, so set it to the final opinion (or it will stay 0)
        final_opinion._a = discounted_opinions[0]._a

        prob = final_opinion.projected_probability()
        self._last_predict_proba = prob

        if self._debug:
            print("\n* Effective weights:")
            all_opinions = [m.discounted_information_opinion for m in self.slmodels]
            if self._base_rate_choice == 1:
                all_trust_opinions = [m._modified_trust for m in self.slmodels]
                most_trusted = find_max_belief(all_trust_opinions)
            else:
                most_trusted = -1

            denominator = 0
            for i in range(len(self.slmodels)):
                denominator += _uncertainty_product(all_opinions, i)
            # Now process each model alone
            for i in range(len(self.slmodels)):
                m = self.slmodels[i]
                weight = m._modified_trust._b*_uncertainty_product(all_opinions, i)/denominator
                if i == most_trusted:
                    weight += final_opinion._u
                print("Model %s: %.3g" % (m.name, weight))
                self._slm_weights[i].append(weight)

            # Case of prior mode, the previous opinion has its weight
            if self._base_rate_choice == 0:
                print("Previous prediction: %.3g" % final_opinion._u)
                self._slm_weights[len(self.slmodels)].append(final_opinion._u)

            print("\n* Final opinion:", final_opinion)
            # Projected probability is made of 2 parts: belief and contribution of prior probability
            print("Base rate contribution = %.3g" % (final_opinion._a * final_opinion._u))
            print("Class 1 Probability = %.3g" % prob)

            self._slm_uncertainty[len(self.slmodels)].append(final_opinion._u)

        # Flow ID awareness when needed
        if self._multi_flow:
            self._last_id = self._id_list[self._cache_i]

        return prob

    def _gen_prediction_cache(self, samples: pd.DataFrame):
        """Fills the prediction cache of all models for the current samples"""
        # When in multi flow mode, we track the trust penalties and previous final opinion per flow
        if self._multi_flow:
            self._id_list = array('Q', samples[self._id_col].array)

        for model in self.slmodels:
            model.predict_proba_to_cache(samples)
        self._cache_i = 0
        self._cache_max = samples.shape[0]

    def _run_once(self) -> float:
        if self._debug:
            print("--------------------------------------------")
            print("Iteration", self._cache_i)

        if self._multi_flow:
            current_id = self._id_list[self._cache_i]
            if current_id == self._last_id and self._base_rate_choice == 0:
                # The flow ID didn't change from the last iteration
                # And our base rate source is the last prediction probability
                # We only need to update the base rate in this case
                self._set_all_base_rates(self._last_predict_proba)
            else:
                self._save_state(self._last_id)
                self._load_state(current_id)
        elif self._base_rate_choice == 0:
            self._set_all_base_rates(self._last_predict_proba)

        # Estimate opinions and their modified average (assuming all inference is done earlier)
        self._get_all_opinions_and_ref()
        # Find conflict between undiscounted opinions and modified average
        self._get_all_conflicts()
        # Based on conflicts, find new trust values
        self._reevaluate_trust()
        return self._get_final_prediction()

    def auto_tune(self, samples, true_labels, bonus_step=0.2, descending_order=True, over_stepping=False, _show_progress=False):
        """
        Finds good trust bonuses for the given dataset. Sets models trust opinion from their MCC.

        Parameters
        ----------
        samples: Dataset sample points

        true_labels: The true labels corresponding to samples, should be binary.

        bonus_step: Determines the step size while modifying bonuses.

        descending_order: When set to True, models are processed in the order of decreasing MCC

        over_stepping: If True, the function keeps trying with higher bonus steps, may provide better bonuses with longer runtime.

        Note
        ----
        The absolute value of bonuses is capped by the maximum penalty.
        """
        if bonus_step < 0.05:
            raise ValueError("Bonus value %g is too small, it should be greater than 0.05")

        self.trust_from_dataset_mcc(samples, true_labels)
        # Reset bonus for all models and set initial trust from MCC
        for slmodel in self.slmodels:
            slmodel.set_bonuses(0, 0)

        # Sort models in the internal list according to the trust
        self.slmodels.sort(key=lambda x: x._trust_opinion._b, reverse=descending_order)

        # Perform a run without bonuses to get a baseline of models behavior under conflict
        predicted = self.predict(samples, True, true_labels)
        old_mcc = matthews_corrcoef(true_labels, predicted)

        if _show_progress:
            print("Baseline MCC (no bonuses): %g" % old_mcc)

        max_bonus = self.max_penalty

        # Traversing models in descending order
        for model in self.slmodels:
            if _show_progress:
                print("Tuning bonuses of model \"%s\" started:" % model.name)
            # Loop to find the best positive class bonus
            if model.pcumulative_conflict > 0:
                old_bonus = 0
                max_reached = False
                curr_step = bonus_step
                # Increase/decrease the bonus while monitoring MCC
                cicr_0 = model.pconflict_TP/model.pcumulative_conflict
                dist = cicr_0 - 0.5
                while True:
                    if dist > 0:
                        model.pclass_bonus = min(max_bonus, old_bonus+curr_step)
                    else:
                        model.pclass_bonus = max(-max_bonus, old_bonus-curr_step)

                    if abs(model.pclass_bonus) == max_bonus:
                        max_reached = True

                    predicted = self.predict(samples, True, true_labels)
                    new_mcc = matthews_corrcoef(true_labels, predicted)
                    # If our increment/decrement didn't provide improvements, roll it back
                    if new_mcc < old_mcc:
                        model.pclass_bonus = old_bonus
                        if over_stepping:
                            curr_step *= 2
                        else:
                            break
                    else:
                        old_bonus = model.pclass_bonus
                        old_mcc = new_mcc

                    # Needed the flag to be set earlier to prevent flapping between two bonus values in case new_mcc < old_mcc
                    if max_reached:
                        break

                if _show_progress:
                    print("Class 1 bonus = %g, CICR = %g, MCC = %g" % (model.pclass_bonus, cicr_0 , old_mcc))

            # The same algorithm but for the negative class bonus
            if model.ncumulative_conflict > 0:
                old_bonus = 0
                max_reached = False
                curr_step = bonus_step
                # Increase/decrease the bonus while monitoring MCC
                cicr_1 = model.nconflict_TN/model.ncumulative_conflict
                dist = cicr_1 - 0.5
                while True:
                    if dist > 0:
                        model.nclass_bonus = min(max_bonus, old_bonus+curr_step)
                    else:
                        model.nclass_bonus = max(-max_bonus, old_bonus-curr_step)

                    if abs(model.nclass_bonus) == max_bonus:
                        max_reached = True

                    model.nclass_bonus = model.nclass_bonus
                    predicted = self.predict(samples, True, true_labels)
                    new_mcc = matthews_corrcoef(true_labels, predicted)
                    # If our increment/decrement didn't provide improvements, roll it back
                    if new_mcc < old_mcc:
                        model.nclass_bonus = old_bonus
                        if over_stepping:
                            curr_step *= 2
                        else:
                            break
                    else:
                        old_bonus = model.nclass_bonus
                        old_mcc = new_mcc

                    # Needed the flag to be set earlier to prevent flapping between two bonus values in case new_mcc < old_mcc
                    if max_reached:
                        break

                if _show_progress:
                    print("Class 0 bonus = %g, CICR = %g, MCC = %g" % (model.nclass_bonus, cicr_1 , old_mcc))

    def _predict_proba(self, X, _keep_caches=False, _true_labels=None) -> np.ndarray:
        """Predict using the ensemble of models

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

        if self._debug:
            count = len(self.slmodels)+int(self._debug)
            # If debugging, we want to store the distance to average conflicts, penalties and weights
            self._slm_dist_to_avg = [array('f') for _ in range(count)]
            self._slm_weights = [array('f') for _ in range(count)]
            self._slm_penalties = [array('f') for _ in range(count)]
            self._slm_uncertainty = [array('f') for _ in range(count+1)]
        else:
            # Otherwise flush older runs
            self._slm_dist_to_avg = self._slm_weights = self._slm_penalties = []

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
            results[input_row] = class1

        return np.asarray(results)

    def predict(self, X, _keep_caches=False, _true_labels=None):
        """Predict using the ensemble of models added.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples)
        """
        return self._predict_proba(X, _keep_caches, _true_labels).round()

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

    def _soft_vote_prob(self):
        caches = self._merge_caches()
        sum_votes = np.sum(caches, axis=1)
        prob = np.divide(sum_votes, len(self.slmodels))
        return prob

    def _soft_vote(self):
        prob = self._soft_vote_prob()
        return np.round(prob)
