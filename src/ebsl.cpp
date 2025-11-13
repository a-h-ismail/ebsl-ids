#include "binomial_opinion.h"
#include "bsl_sm.h"
#include "ebsl.h"

EBSL::EBSL(float conflict_threshold, float max_penalty, float b, float trust_restore_speed, int base_rate_choice)
{
    enable_debugging = false;
    multi_flow = false;
    all_discounted_opinions.reserve(9);
    all_conflicts.reserve(9);
    EBSL::conflict_threshold = conflict_threshold;
    EBSL::max_penalty = max_penalty;
    EBSL::b = b;
    EBSL::trust_restore_speed = trust_restore_speed;
    EBSL::base_rate_choice = base_rate_choice;
    last_prediction = 0;
}

void EBSL::clear_sl_state()
{
    last_prediction = 0.49;
    set_all_base_rates(last_prediction);
    // Reset all conflict related counters and the modified trust opinion
    for (BSL_SM *m : slmodels)
    {
        m->conflict = m->conflict_counter = m->curr_bonus = m->trust_offset = 0;
        m->modified_trust = m->trust;
    }
}

std::string EBSL::to_string()
{
    char tmp[200];
    const char *base_rate_choice_str;
    switch (base_rate_choice)
    {
    case PRIOR_SOURCE:
        base_rate_choice_str = "prior";
        break;
    case TRUST_SOURCE:
        base_rate_choice_str = "trust";
        break;
    default:
        // A nice warning to those who forget to update this switch
        base_rate_choice_str = "UNKNOWN!!";
    }
    sprintf(tmp, "EBSL classifier: conflict_threshold=%g, max_penalty=%g, b=%g, trust_restore_speed=%g, base_rate_choice:\"%s\", nb_of_classifiers = %d",
            conflict_threshold, max_penalty, b, trust_restore_speed, base_rate_choice_str, (int)slmodels.size());
    return std::string(tmp);
}

void EBSL::add_model(BSL_SM *model)
{
    // if model->name in _slmodels_dict:
    //     raise ValueError("The Model with name %s already exists!" % model->name)
    slmodels.push_back(model);
    //_slmodels_dict[model->name] = model
    // Clear already stored states just in case
    state_store.clear();
}

// Returns the index of the highest belief in modified trust opinions
int EBSL::find_most_trusted()
{
    int max = -1, max_index = -1;
    for (int i = 0; i < slmodels.size(); ++i)
        if (max < slmodels[i]->modified_trust.b)
        {
            max_index = i;
            max = slmodels[i]->modified_trust.b;
        }

    return max_index;
}

void EBSL::set_all_base_rates(float base_rate)
{
    // Set the base rate for all original information opinions
    for (int i = 0; i < slmodels.size(); ++i)
        slmodels[i]->information.a = base_rate;
}

void EBSL::save_state(int64_t flow_id)
{
    try
    {
        auto state_entry = state_store.at(flow_id);
        // Update the stored state instead of allocating a new one with updated information
        state_entry.last_prediction_proba = last_prediction;

        // Store updated bonuses and conflict counters
        for (int i = 0; i < slmodels.size(); ++i)
        {
            state_entry.conflict_counters[i] = slmodels[i]->conflict_counter;
            state_entry.bonuses[i] = slmodels[i]->curr_bonus;
            state_entry.trust_offset[i] = slmodels[i]->trust_offset;
        }
        state_store[flow_id] = state_entry;
    }
    // Entry does not exist, create it
    catch (std::out_of_range e)
    {
        sl_state_snapshot new_entry;
        new_entry.last_prediction_proba = last_prediction;
        // Store bonuses and conflict counters
        for (int i = 0; i < slmodels.size(); ++i)
        {
            new_entry.conflict_counters.push_back(slmodels[i]->conflict_counter);
            new_entry.bonuses.push_back(slmodels[i]->curr_bonus);
            new_entry.trust_offset.push_back(slmodels[i]->trust_offset);
        }
        state_store[flow_id] = new_entry;
    }
}

void EBSL::load_state(int64_t flow_id)
{
    try
    {
        auto state_entry = state_store.at(flow_id);
        // Restore the last prediction
        last_prediction = state_entry.last_prediction_proba;

        // Load updated bonuses and conflict counters and recalculate the modified trust
        for (int i = 0; i < slmodels.size(); ++i)
        {
            slmodels[i]->conflict_counter = state_entry.conflict_counters[i];
            slmodels[i]->curr_bonus = state_entry.bonuses[i];
            slmodels[i]->trust_offset = state_entry.trust_offset[i];
            modify_trust(slmodels[i]->trust, slmodels[i]->trust_offset, slmodels[i]->modified_trust);
        }
        set_all_base_rates(last_prediction);
    }
    // Use a default initial state
    catch (std::out_of_range e)
    {
        clear_sl_state();
    }
}

float EBSL::get_penalty(float nb_conflict)
{
    return max_penalty * nb_conflict / (nb_conflict + b);
}

void EBSL::get_all_information_opinions()
{
    // Gets all original model opinions (from the state cache) then calculates the reference opinion if necessary

    for (BSL_SM *slmodel : slmodels)
        slmodel->get_information_opinion(iteration);

    // Case where the base rate strategy was: highest trust
    // Find model with the current highest trust and use its belief as the base rate

    if (base_rate_choice == TRUST_SOURCE)
    {
        int highest_trust_index = find_most_trusted();
        set_all_base_rates(slmodels[highest_trust_index]->information.b);
    }

    // Calculate the information opinion after discounting with modified trust
    for (BSL_SM *slmodel : slmodels)
        slmodel->get_discounted_information_opinion();

    if (enable_debugging)
    {
        puts("* Initialization");
        puts("Information opinions:");
        for (BSL_SM *slmodel : slmodels)
        {
            printf("Model %s: ", slmodel->name.c_str());
            slmodel->information.print_opinion();
            putchar('\n');
        }

        puts("\n* Before trust update");
        puts("Discounted information opinions:");
        for (BSL_SM *slmodel : slmodels)
        {
            printf("Model %s: ", slmodel->name.c_str());
            slmodel->discounted_information.print_opinion();
            putchar('\n');
        }
    }
}

// Calculate conflict relative to the reference opinion. Results are stored in each model object
void EBSL::get_ref_and_conflicts()
{
    if (base_rate_choice == PRIOR_SOURCE)
    {
        // Remember that the base rate is the last prior probability
        float a = slmodels[0]->information.a;
        reference.b = a;
        reference.d = 1 - a;
        reference.u = 0;
    }
    else if (base_rate_choice == TRUST_SOURCE)
    {
        // Calculate the average of discounted opinions
        all_discounted_opinions.clear();
        for (BSL_SM *model : slmodels)
            all_discounted_opinions.push_back(model->discounted_information);
        reference = average_fusion(all_discounted_opinions);
    }

    if (enable_debugging)
    {
        printf("Reference opinion: ");
        reference.print_opinion();
        puts("\n");
    }

    // Calculate conflict of each model with the reference
    for (BSL_SM *slmodel : slmodels)
        slmodel->conflict = slmodel->information.calculate_conflict(reference);
}

// Updates the trust for each model according to the conflict
void EBSL::reevaluate_trust()
{
    float total_conflict = 0, average_conflict;
    for (BSL_SM *model : slmodels)
        total_conflict += model->conflict;
    average_conflict = total_conflict / slmodels.size();

    for (BSL_SM *slmodel : slmodels)
    {
        float distance_to_average_conf;
        distance_to_average_conf = slmodel->conflict - average_conflict;

        if (distance_to_average_conf > conflict_threshold)
        {
            // Condition met, increment the conflict counter to increase penalty
            ++slmodel->conflict_counter;
            slmodel->trust_offset = get_penalty(slmodel->conflict_counter);

            // The model is predicting the positive class, so use the pclass bonus
            if (slmodel->information.b >= 0.5)
            {
                slmodel->pcumulative_conflict += 1;
                // Used for bonus tuning (to know which models are worth giving bonuses)
                if (compare_to_true_labels && true_labels.data()[iteration] == true)
                    ++slmodel->pconflict_TP;

                slmodel->trust_offset -= slmodel->pclass_bonus;
                slmodel->curr_bonus = slmodel->pclass_bonus;
            }
            else
            {
                // Predicting the negative class
                ++slmodel->ncumulative_conflict;
                // Also for bonus tuning
                if (compare_to_true_labels && true_labels.data()[iteration] == false)
                    ++slmodel->nconflict_TN;

                slmodel->trust_offset -= slmodel->nclass_bonus;
                slmodel->curr_bonus = slmodel->nclass_bonus;
            }
            // Recalculate the modified trust opinion immediately (to avoid EBSL::always recalculating all trust opinions later)
            modify_trust(slmodel->trust, slmodel->trust_offset, slmodel->modified_trust);
            // And the new discounted information opinion
            slmodel->get_discounted_information_opinion();
        }
        // Not enough conflict, start restoring trust if needed
        else if (slmodel->conflict_counter != 0)
        { // Gradually restoring trust
            slmodel->conflict_counter -= std::min(trust_restore_speed, slmodel->conflict_counter);

            if (slmodel->conflict_counter == 0)
                // Reset both counters
                slmodel->trust_offset = slmodel->curr_bonus = 0;
            else
                slmodel->trust_offset = get_penalty(slmodel->conflict_counter) - slmodel->curr_bonus;

            // recalculate trust and discounted information opinions
            modify_trust(slmodel->trust, slmodel->trust_offset, slmodel->modified_trust);
            slmodel->get_discounted_information_opinion();
        }
    }
    if (enable_debugging)
    {
        puts("* Conflict statistics");
        printf("Conflict:");
        for (BSL_SM *model : slmodels)
            printf("%.3g ", model->conflict);

        printf("\nAverage conflict = %.3g\n", average_conflict);
        printf("Distance to average: ");
        for (BSL_SM *model : slmodels)
            printf("%.3g ", model->conflict - average_conflict);

        puts("\n");
        for (BSL_SM *m : slmodels)
            printf("Model %s: Conflict count = %g, curr_penalty = %.3g, bonus = %.3g\n",
                   m->name.c_str(), m->conflict_counter, get_penalty(m->conflict_counter), m->curr_bonus);

        puts("\n* After trust update");
        puts("Discounted information opinions:");
        for (BSL_SM *slmodel : slmodels)
        {
            printf("Model %s: ", slmodel->name.c_str());
            slmodel->discounted_information.print_opinion();
            putchar('\n');
        }

        // Store dist to average conflict, penalties and uncertainty
        for (int i = 0; i < slmodels.size(); ++i)
        {
            slm_dist_to_avg[i].push_back(slmodels[i]->conflict - average_conflict);
            slm_penalties[i].push_back(slmodels[i]->trust_offset);
            slm_uncertainty[i].push_back(slmodels[i]->discounted_information.u);
        }
    }
}

// Calculates the final prediction using discounted information opinion
float EBSL::get_final_prediction()
{
    all_discounted_opinions.clear();
    for (BSL_SM *slmodel : slmodels)
        all_discounted_opinions.push_back(slmodel->discounted_information);

    last_final_opinion = average_fusion(all_discounted_opinions);
    // The base rate should be the same everywhere, so set it to the final opinion (or it will stay 0)
    last_final_opinion.a = all_discounted_opinions[0].a;

    last_prediction = last_final_opinion.projected_probability();

    if (enable_debugging)
    {
        puts("\n* Effective weights:");
        int most_trusted;
        if (base_rate_choice == TRUST_SOURCE)
            most_trusted = find_most_trusted();
        else
            most_trusted = -1;

        float denominator = 0;
        for (int i = 0; i < slmodels.size(); ++i)
            denominator += uncertainty_product(all_discounted_opinions, i);

        // Now calculate the weight of each model
        for (int i = 0; i < slmodels.size(); ++i)
        {
            float weight = slmodels[i]->modified_trust.b * uncertainty_product(all_discounted_opinions, i) / denominator;
            if (i == most_trusted)
                weight += last_final_opinion.u;
            printf("Model %s: %.3g\n", slmodels[i]->name.c_str(), weight);
            slm_weights[i].push_back(weight);
        }

        // Case of prior mode, the previous opinion has its weight
        if (base_rate_choice == PRIOR_SOURCE)
            printf("Previous prediction: %.3g\n", last_final_opinion.u);
        slm_weights[slmodels.size()].push_back(last_final_opinion.u);

        printf("\n* Final opinion:");
        last_final_opinion.print_opinion();
        // Projected probability is made of 2 parts: belief and contribution of prior probability
        printf("\nBase rate contribution = %.3g\n", last_final_opinion.a * last_final_opinion.u);
        printf("Class 1 Probability = %.3g\n", last_prediction);

        slm_uncertainty[slmodels.size()].push_back(last_final_opinion.u);
    }

    // Flow ID awareness when needed
    if (multi_flow)
        last_id = id_list.data()[iteration];

    return last_prediction;
}

float EBSL::run_once()
{
    if (enable_debugging)
    {
        puts("--------------------------------------------");
        printf("Iteration %d\n", iteration);
    }

    if (multi_flow)
    {
        int64_t current_id = id_list.data()[iteration];
        if (current_id == last_id && base_rate_choice == PRIOR_SOURCE)
            // The flow ID didn't change from the last iteration
            // And our base rate source is the last prediction probability
            // We only need to update the base rate in this case
            set_all_base_rates(last_prediction);
        else
        {
            save_state(last_id);
            load_state(current_id);
        }
    }
    else if (base_rate_choice == 0)
        set_all_base_rates(last_prediction);

    // Estimate opinions and their modified average (assuming all inference is done earlier)
    get_all_information_opinions();
    // Find conflict between undiscounted opinions and modified average
    get_ref_and_conflicts();
    // Based on conflicts, find new trust values
    reevaluate_trust();
    return get_final_prediction();
}
// Run the ebsl algorithm on already stored predictions
// If true_labels is also provided, collects CICR statistics

void EBSL::predict_proba()
{
    /*Predict using the ensemble of models

    Parameters
    ----------
    X : Dataframe of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    y : ndarray, shape (n_samples)
    */

    // Clear previous runs
    slm_dist_to_avg.clear();
    slm_weights.clear();
    slm_penalties.clear();
    slm_uncertainty.clear();

    if (enable_debugging)
    {
        int count = slmodels.size();
        std::vector<float> tmp;
        // If debugging, we want to store the distance to average conflicts, penalties and weights
        for (int i = 0; i < count; ++i)
        {
            // tmp is sent by value
            slm_dist_to_avg.push_back(tmp);
            slm_weights.push_back(tmp);
            slm_uncertainty.push_back(tmp);
            slm_penalties.push_back(tmp);
        }
        if (base_rate_choice == PRIOR_SOURCE)
        {
            // Account for the weight and uncertainty of the last prediction in prior mode
            slm_weights.push_back(tmp);
            slm_uncertainty.push_back(tmp);
        }
    }

    // Clear the state map
    state_store.clear();

    // Clear old conflict statistics
    for (BSL_SM *m : slmodels)
    {
        m->pcumulative_conflict = m->pconflict_TP = 0;
        m->ncumulative_conflict = m->nconflict_TN = 0;
    }

    int nb_rows = class1_prediction.shape(0);

    float class1;
    for (iteration = 0; iteration < nb_rows; ++iteration)
    {
        class1 = run_once();
        class1_prediction.data()[iteration] = class1;
    }
}

int ebsl_tp_traverse(PyObject *self, visitproc visit, void *arg)
{
// On Python 3.9+, we must traverse the implicit dependency
// of an object on its associated type object.
#if PY_VERSION_HEX >= 0x03090000
    Py_VISIT(Py_TYPE(self));
#endif

    // The tp_traverse method may be called after __new__ but before or during
    // __init__, before the C++ constructor has been completed. We must not
    // inspect the C++ state if the constructor has not yet completed.
    if (!nb::inst_ready(self))
    {
        return 0;
    }

    // Get the C++ object associated with 'self' (this always succeeds)
    EBSL *w = nb::inst_ptr<EBSL>(self);

    // If w->value has an associated Python object, return it.
    // If not, value.ptr() will equal NULL, which is also fine.
    // We need to inform the python garbage collector that we have numpy arrays here
    Py_VISIT(nb::find(w->id_list).ptr());
    Py_VISIT(nb::find(w->class1_prediction).ptr());
    Py_VISIT(nb::find(w->true_labels).ptr());
    return 0;
}

int ebsl_tp_clear(PyObject *self)
{
    // Get the C++ object associated with 'self' (this always succeeds)
    EBSL *w = nb::inst_ptr<EBSL>(self);

    // TODO: Is this necessary?
    w->slmodels.clear();

    return 0;
}

// Table of custom type slots we want to install
PyType_Slot wrapper_slots[] = {
    {Py_tp_traverse, (void *)ebsl_tp_traverse},
    {Py_tp_clear, (void *)ebsl_tp_clear},
    {0, 0}};

NB_MODULE(ebsl_cpp, m)
{
    nb::class_<EBSL>(m, "EBSL_cpp", nb::type_slots(wrapper_slots))
        .def(nb::init<>())
        .def(nb::init<float, float, float, float, int>(), "conflict_threshold"_a = 0.15, "max_penalty"_a = 0.5, "b"_a = 1., "trust_restore_speed"_a = 0.5, "base_rate_choice"_a = (int)PRIOR_SOURCE)
        .def("__str__", &EBSL::to_string)
        .def("add_model", &EBSL::add_model, "model"_a)
        .def("predict_proba", &EBSL::predict_proba)
        .def_rw("max_penalty", &EBSL::max_penalty)
        .def_rw("b", &EBSL::b)
        .def_rw("id_list", &EBSL::id_list)
        .def_rw("true_labels", &EBSL::true_labels)
        .def_rw("slmodels", &EBSL::slmodels)
        .def_rw("slm_dist_to_avg", &EBSL::slm_dist_to_avg)
        .def_rw("slm_uncertainty", &EBSL::slm_uncertainty)
        .def_rw("slm_penalties", &EBSL::slm_penalties)
        .def_rw("slm_weights", &EBSL::slm_weights)
        .def_rw("multi_flow", &EBSL::multi_flow)
        .def_rw("compare_to_true_labels", &EBSL::compare_to_true_labels)
        .def_rw("class1_prediction", &EBSL::class1_prediction)
        .def_rw("enable_debugging", &EBSL::enable_debugging)
        .def_rw("conflict_threshold", &EBSL::conflict_threshold)
        .def_rw("base_rate_choice", &EBSL::base_rate_choice)
        .def_rw("iteration", &EBSL::iteration);

    nb::class_<BSL_SM>(m, "BSL_SM_cpp")
        .def(nb::init<>())
        .def("trust_from_mcc", &BSL_SM::trust_from_mcc, "mcc"_a, "w"_a = 2)
        .def("set_bonuses", &BSL_SM::set_bonuses, "class_0"_a, "class_1"_a)
        .def("set_initial_trust_opinion", &BSL_SM::set_initial_trust_opinion, "b"_a, "d"_a, "u"_a)
        .def_rw("prediction_cache", &BSL_SM::prediction_cache)
        .def_rw("trust", &BSL_SM::trust)
        .def_rw("modified_trust", &BSL_SM::modified_trust)
        .def_rw("pclass_bonus", &BSL_SM::pclass_bonus)
        .def_rw("nclass_bonus", &BSL_SM::nclass_bonus)
        .def_rw("pcumulative_conflict", &BSL_SM::pcumulative_conflict)
        .def_rw("pconflict_TP", &BSL_SM::pconflict_TP)
        .def_rw("ncumulative_conflict", &BSL_SM::ncumulative_conflict)
        .def_rw("nconflict_TN", &BSL_SM::nconflict_TN)
        .def_rw("name", &BSL_SM::name);

    m.def("average_fusion", &average_fusion);
    m.def("modify_trust", &modify_trust);
    m.def("uncertainty_product", &uncertainty_product);
    nb::class_<Opinion>(m, "Opinion")
        .def(nb::init<>())
        .def(nb::init<float, float, float, float>(), "b"_a, "d"_a, "u"_a, "a"_a = 1)
        .def("__str__", &Opinion::to_string)
        .def("validate_opinions", &Opinion::validate_opinion)
        .def("set_parameters", &Opinion::set_parameters, "b"_a, "d"_a, "u"_a, "a"_a = 1)
        .def("calculate_conflict", &Opinion::calculate_conflict, "reference"_a)
        .def("trust_discounting", &Opinion::trust_discounting, "trust"_a, "out"_a)
        .def("projected_probability", &Opinion::projected_probability)
        .def("print_opinion", &Opinion::print_opinion)
        .def_rw("b", &Opinion::b)
        .def_rw("d", &Opinion::d)
        .def_rw("u", &Opinion::u)
        .def_rw("a", &Opinion::a);
}
