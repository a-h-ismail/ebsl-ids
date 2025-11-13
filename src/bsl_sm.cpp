#include "binomial_opinion.h"
#include "bsl_sm.h"
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;
using namespace nb::literals;

// Resets the subjective logic state of this model
void BSL_SM::reset_sl_state()
{
    modified_trust = trust;
    information = discounted_information = Opinion();
    conflict = 0;
    conflict_counter = 0;
    pcumulative_conflict = pconflict_TP = ncumulative_conflict = nconflict_TN = 0;
    curr_bonus = 0;
}

void BSL_SM::set_initial_trust_opinion(float b, float d, float u)
{
    trust.set_parameters(b, d, u);
    modified_trust.set_parameters(b, d, u);
}

void BSL_SM::trust_from_mcc(float mcc, int w)
{
    float scale = (float)100 / (100 + w);
    set_initial_trust_opinion(mcc * scale, (1 - mcc) * scale, 1 - scale);
}

void BSL_SM::set_bonuses(float class0_bonus, float class1_bonus)
{
    pclass_bonus = class0_bonus;
    nclass_bonus = class1_bonus;
}

float BSL_SM::get_prediction(int index)
{
    return prediction_cache.data()[index];
}

void BSL_SM::get_information_opinion(int current_index)
{
    float p = get_prediction(current_index);
    information.b = p;
    information.d = 1 - p;
}

/*
Calculates the discounted opinion according to the trust opinion modified with trust penalty and bonus.

Warning: This function assumes you already called get_information_opinion() and updated the modified trust if necessary
*/
void BSL_SM::get_discounted_information_opinion()
{
    information.trust_discounting(modified_trust, discounted_information);
}
