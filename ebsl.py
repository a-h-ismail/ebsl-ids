#!/usr/bin/env python3
from array import array
import copy
from typing import Literal
from uuid import uuid4
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from ebsl_cpp import *


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

        if name == "":
            name = str(uuid4()).replace('-', '')[:16]

        # The C++ core implementation
        self.bsl_cpp = BSL_SM_cpp()
        if trust_opinion is not None:
            self.bsl_cpp.trust = trust_opinion

        self.name = name

    # For direct access of the C++ implementation from Python
    @property
    def trust(self):
        return self.bsl_cpp.trust

    @trust.setter
    def trust(self, value):
        self.bsl_cpp.trust = value

    @property
    def prediction_cache(self):
        return self.bsl_cpp.prediction_cache

    @prediction_cache.setter
    def prediction_cache(self, value):
        self.bsl_cpp.prediction_cache = value

    @property
    def pclass_bonus(self):
        return self.bsl_cpp.pclass_bonus

    @pclass_bonus.setter
    def pclass_bonus(self, value):
        self.bsl_cpp.pclass_bonus = value

    @property
    def nclass_bonus(self):
        return self.bsl_cpp.nclass_bonus

    @nclass_bonus.setter
    def nclass_bonus(self, value):
        self.bsl_cpp.nclass_bonus = value

    @property
    def pcumulative_conflict(self):
        return self.bsl_cpp.pcumulative_conflict

    @property
    def ncumulative_conflict(self):
        return self.bsl_cpp.ncumulative_conflict

    @property
    def pconflict_TP(self):
        return self.bsl_cpp.pconflict_TP

    @property
    def nconflict_TN(self):
        return self.bsl_cpp.nconflict_TN

    @property
    def name(self):
        return self.bsl_cpp.name

    @name.setter
    def name(self, value):
        self.bsl_cpp.name = value

    def set_initial_trust_opinion(self, b, d, u):
        self.bsl_cpp.set_initial_trust_opinion(b, d, u)

    def trust_from_mcc(self, mcc: float, w=2):
        """Sets the trust opinion of this model using its Matthews correlation coefficient (MCC)"""
        self.bsl_cpp.trust_from_mcc(mcc, w)

    def set_bonuses(self, nclass_bonus: float, pclass_bonus: float):
        self.bsl_cpp.set_bonuses(nclass_bonus, pclass_bonus)

    def predict_proba_to_cache(self, samples):
        """
        Calls predict_proba of the underlying model after scaling the input samples
        """
        if self.scaler is not None:
            samples = samples[self.scaler.get_feature_names_out()]
            samples = self.scaler.transform(samples)

        self._pycache_holder = np.asarray(self.model.predict_proba(samples)[:, 1], dtype=np.float32, order='C')
        self.prediction_cache = self._pycache_holder


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
        self.base_rate_choice: str
        self.slmodels: list[BSL_SM] = []
        if base_rate_choice == "prior":
            base_rate_choice_val = 0
        elif base_rate_choice == "trust":
            base_rate_choice_val = 1

        self.base_rate_choice = base_rate_choice
        self._id_col = id_col
        self.ebsl_cpp = EBSL_cpp(conflict_threshold, max_penalty, b, trust_restore_speed, base_rate_choice_val)
        self.ebsl_cpp.enable_debugging = _debug
        self._slmodels_dict = {}

    def __str__(self) -> str:
        return self.ebsl_cpp.__str__()

    def get_model_by_name(self, name: str) -> BSL_SM:
        return self._slmodels_dict[name]

    def add_model(self, model: BSL_SM):
        if model.name in self._slmodels_dict:
           raise RuntimeError("The Model with name %s already exists!" % model.name)
        else:
            self.slmodels.append(model)
            self.ebsl_cpp.add_model(model.bsl_cpp)
            self._slmodels_dict[model.name] = model

    def _gen_prediction_cache(self, samples: pd.DataFrame):
        """Fills the prediction cache of all models for the current samples"""
        for model in self.slmodels:
            model.predict_proba_to_cache(samples)

    def trust_from_dataset_mcc(self, samples: pd.DataFrame, true_labels) -> None:
        """Set the trust of all Models in the ensemble from the MCC metrics of the provided dataset
        Note: This function fills the prediction cache for all models."""
        self._gen_prediction_cache(samples)
        true_labels = np.asarray(true_labels)
        for model in self.slmodels:
            # pyright: ignore[reportCallIssue, reportArgumentType]
            mcc = matthews_corrcoef(true_labels, np.round(model.prediction_cache))
            model.trust_from_mcc(mcc)

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
        self.slmodels.sort(key=lambda x: x.bsl_cpp.trust.b, reverse=descending_order)

        # Perform a run without bonuses to get a baseline of models behavior under conflict
        predicted = self.predict(samples, False, true_labels)
        old_mcc = matthews_corrcoef(true_labels, predicted)

        if _show_progress:
            print("Baseline MCC (no bonuses): %g" % old_mcc)

        max_bonus = self.ebsl_cpp.max_penalty

        # Traversing models in descending order
        for model in self.slmodels:
            if _show_progress:
                print("Tuning bonuses of model \"%s\" started:" % model.bsl_cpp.name)
            # Loop to find the best positive class bonus
            if model.bsl_cpp.pcumulative_conflict > 0:
                old_bonus = 0
                max_reached = False
                curr_step = bonus_step
                # Increase/decrease the bonus while monitoring MCC
                cicr_0 = model.bsl_cpp.pconflict_TP/model.bsl_cpp.pcumulative_conflict
                dist = cicr_0 - 0.5
                while True:
                    if dist > 0:
                        model.bsl_cpp.pclass_bonus = min(max_bonus, old_bonus+curr_step)
                    else:
                        model.bsl_cpp.pclass_bonus = max(-max_bonus, old_bonus-curr_step)

                    if abs(model.bsl_cpp.pclass_bonus) == max_bonus:
                        max_reached = True

                    predicted = self.predict(samples, True, true_labels)
                    new_mcc = matthews_corrcoef(true_labels, predicted)
                    # If our increment/decrement didn't provide improvements, roll it back
                    if new_mcc < old_mcc:
                        model.bsl_cpp.pclass_bonus = old_bonus
                        if over_stepping:
                            curr_step *= 2
                        else:
                            break
                    else:
                        old_bonus = model.bsl_cpp.pclass_bonus
                        old_mcc = new_mcc

                    # Needed the flag to be set earlier to prevent flapping between two bonus values in case new_mcc < old_mcc
                    if max_reached:
                        break

                if _show_progress:
                    print("Class 1 bonus = %g, CICR = %g, MCC = %g" % (model.bsl_cpp.pclass_bonus, cicr_0, old_mcc))

            # The same algorithm but for the negative class bonus
            if model.bsl_cpp.ncumulative_conflict > 0:
                old_bonus = 0
                max_reached = False
                curr_step = bonus_step
                # Increase/decrease the bonus while monitoring MCC
                cicr_1 = model.bsl_cpp.nconflict_TN/model.bsl_cpp.ncumulative_conflict
                dist = cicr_1 - 0.5
                while True:
                    if dist > 0:
                        model.bsl_cpp.nclass_bonus = min(max_bonus, old_bonus+curr_step)
                    else:
                        model.bsl_cpp.nclass_bonus = max(-max_bonus, old_bonus-curr_step)

                    if abs(model.bsl_cpp.nclass_bonus) == max_bonus:
                        max_reached = True

                    model.bsl_cpp.nclass_bonus = model.bsl_cpp.nclass_bonus
                    predicted = self.predict(samples, True, true_labels)
                    new_mcc = matthews_corrcoef(true_labels, predicted)
                    # If our increment/decrement didn't provide improvements, roll it back
                    if new_mcc < old_mcc:
                        model.bsl_cpp.nclass_bonus = old_bonus
                        if over_stepping:
                            curr_step *= 2
                        else:
                            break
                    else:
                        old_bonus = model.bsl_cpp.nclass_bonus
                        old_mcc = new_mcc

                    # Needed the flag to be set earlier to prevent flapping between two bonus values in case new_mcc < old_mcc
                    if max_reached:
                        break

                if _show_progress:
                    print("Class 0 bonus = %g, CICR = %g, MCC = %g" % (model.nclass_bonus, cicr_1, old_mcc))

    def cpp_predict(self):
        return self.ebsl_cpp.predict_proba()

    def predict_proba(self, X, _keep_caches=False, _true_labels=None):
        """Predict using the ensemble of models added.

        Parameters
        ----------
        X : Dataframe of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray, shape (n_samples)
        """

        # Inform the C++ side that we have multi_flow information and send the id list over
        if self._id_col != "":
            self.ebsl_cpp.multi_flow = True
            self.ebsl_cpp.id_list = np.asarray(X[self._id_col], dtype=np.int64, order="C")
        else:
            self.ebsl_cpp.multi_flow = False
            self.ebsl_cpp.id_list = np.empty(1, dtype=np.int64, order="C")

        # Generate the prediction cache (overwrites any older existing cache)
        if not _keep_caches:
            self._gen_prediction_cache(X)
        # Set the true labels variable to track CICR values
        if _true_labels is not None:
            self.ebsl_cpp.true_labels = np.asarray(_true_labels, dtype=np.bool, order='C')
            self.ebsl_cpp.compare_to_true_labels = True
        else:
            self.ebsl_cpp.compare_to_true_labels = False

        results = np.empty(len(X), dtype=np.float32, order='C')
        self.ebsl_cpp.class1_prediction = results
        self.cpp_predict()

        return self.ebsl_cpp.class1_prediction

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
        return self.predict_proba(X, _keep_caches, _true_labels).round()

    @property
    def slm_dist_to_avg(self):
        return self.ebsl_cpp.slm_dist_to_avg

    @property
    def slm_weights(self):
        return self.ebsl_cpp.slm_weights

    @property
    def slm_uncertainty(self):
        return self.ebsl_cpp.slm_uncertainty

    @property
    def slm_penalties(self):
        return self.ebsl_cpp.slm_penalties

    @property
    def conflict_threshold(self):
        return self.ebsl_cpp.conflict_threshold

    def _merge_caches(self):
        """Combines all predictions"""
        predictions = []
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
