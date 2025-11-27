// Copyright (C) 2025 Ahmad Ismail
// SPDX-License-Identifier: MPL-2.0

#ifndef BSL_SM_H
#define BSL_SM_H
#include "binomial_opinion.h"
#include <vector>
#include <string>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

class BSL_SM
{
    /*
    Creates the building blocks required for Ensemble Binomial Subjective Logic.
    It manages subjective logic opinions (information, trust, conflict...) for one ML model

    Parameters
    ----------

    trust_opinion: Indicates the trustworthiness of a model. Affects the contribution of each model to the final prediction.

    name: The model name. If none is provided, a random one is generated
    */

    // EBSL needs access to private members of BSL_SM
    friend class EBSL;

private:
    float conflict, conflict_counter;

    float get_prediction(int index);

    void get_information_opinion(int current_index);

    void get_discounted_information_opinion();

public:
    nb::ndarray<float, nb::numpy, nb::shape<-1>, nb::c_contig> prediction_cache;
    float trust_offset;
    Opinion trust, modified_trust;
    Opinion information, discounted_information;
    int pcumulative_conflict, pconflict_TP, ncumulative_conflict, nconflict_TN;
    float nclass_bonus, pclass_bonus, curr_bonus;
    std::string name;

    // Resets the subjective logic state of this model
    void reset_sl_state();

    void set_initial_trust_opinion(float b, float d, float u);

    void trust_from_mcc(float mcc, int w = 2);

    void set_bonuses(float class0_bonus = 0, float class1_bonus = 0);
};

#endif