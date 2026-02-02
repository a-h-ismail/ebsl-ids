// Copyright (C) 2025 Ahmad Ismail
// SPDX-License-Identifier: MPL-2.0

/**
 * @file
 * Declares the BSL_SM class, which maintains the subjective logic state for one machine learning model
 */

#ifndef BSL_SM_H
#define BSL_SM_H
#include "binomial_opinion.h"
#include <nanobind/ndarray.h>
#include <string>

namespace nb = nanobind;

/** Manages the subjective logic state (information, trust, conflict...) for one ML model.
    @note The C++ side is not responsible for the instance of a binary ML model.
          It is the responsibility of the extension user to pass class 1 prediction probabilities as a numpy array to the prediction cache so the C++ side can use them.
    @note Python Bindings expose this class as BSL_SM_cpp
*/
class BSL_SM
{
    // EBSL needs access to private members of BSL_SM
    friend class EBSL;

private:
    /// Conflict value as calculated during the last iteration
    float conflict;

    /// Current conflict counter
    float conflict_counter;

    /// Initializes the information opinion using the prediction probability at the current index
    void get_information_opinion(int64_t current_index);

    /// Initializes the discounted opinion by trust discounting of the information with modified trust
    void get_discounted_information_opinion();

public:
    /// The class 1 predictions produced by the associated ML model. Should be set before trying to run the EBSL algorithm.
    nb::ndarray<float, nb::numpy, nb::shape<-1>, nb::c_contig> prediction_cache;

    /// Current trust offset, calculated as penalty - current bonus
    float trust_offset;

    /// The original trust opinion
    Opinion trust;

    /// Trust opinion after applying trust modification
    Opinion modified_trust;

    /// Original information opinion (class 1, dogmatic)
    Opinion information;

    /// Result of discounting the original opinion using the modified trust opinion
    Opinion discounted_information;

    /// Number of times this model was in conflict while it was predicting class 1
    int64_t pcumulative_conflict;

    /// Number of times this model predicted the label correctly while in conflict and predicting class 1
    int64_t pconflict_TP;

    /// Similar to pcumulative_conflict but for class 0
    int64_t ncumulative_conflict;

    /// Similar to pconflict_TP but for class 0
    int64_t nconflict_TN;

    /// Negative class (class 0) bonus
    float nclass_bonus;

    /// Positive class (class 1) bonus
    float pclass_bonus;

    /// Currently active bonus
    float curr_bonus;

    /// Model name
    std::string name;

    /// Resets the subjective logic state of this model and all statistics
    void reset_sl_state();

    /// Sets the initial trust opinion parameters and automatically updates the modified trust
    void set_initial_trust_opinion(float b, float d, float u);

    /// Sets the initial trust opinion based on the Matthews Correlation Coefficient
    /// @param mcc
    /// @param w Non informative weight, converts some of the MCC contribution to belief into uncertainty. Higher values reserve more uncertainty.
    void trust_from_mcc(float mcc, int w = 2);

    /// Sets the per class bonuses
    void set_bonuses(float class0_bonus = 0, float class1_bonus = 0);
};

#endif