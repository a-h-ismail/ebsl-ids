#include "binomial_opinion.h"
#include "bsl_sm.h"
#include "ebsl.h"
#include <format>

void EBSL::update_discounted_info_vector()
{
    for (int i = 0; i < nb_models; ++i)
        all_discounted_opinions[i] = slmodels[i]->discounted_information;
}

EBSL::EBSL(float conflict_threshold, float max_penalty, float b, float trust_restore_speed, int base_rate_choice)
{
    enable_debugging = false;
    multi_flow = false;
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
    models_in_conflict = 0;
}

std::string EBSL::to_string()
{
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
        throw std::runtime_error(std::format("Unrecognized operation mode \"{}\"", base_rate_choice));
    }
    return std::format("EBSL classifier: conflict_threshold={:g}, max_penalty={:g}, b={:g}, trust_restore_speed={:g}, base_rate_choice:\"{}\", nb_of_classifiers = {}",
                       conflict_threshold, max_penalty, b, trust_restore_speed, base_rate_choice_str, (int)nb_models);
}

void EBSL::add_model(BSL_SM *model)
{
    try
    {
        slmodels_map.at(model->name);
        // If the code continued here, the model already exists. Raise an exception
        throw std::runtime_error(std::format("The model with name {} already exists!", model->name));
    }
    catch (const std::out_of_range &e)
    {
        // As it should be, the name doesn't exist so add this model
        slmodels.push_back(model);
        slmodels_map[model->name] = model;
        // Clear already stored states just in case
        states_map.clear();
        nb_models = slmodels.size();
        all_discounted_opinions.resize(nb_models);
    }
}

int EBSL::find_most_trusted()
{
    float max = -1;
    int max_index = -1;
    for (int i = 0; i < nb_models; ++i)
        if (max < slmodels[i]->modified_trust.b)
        {
            max_index = i;
            max = slmodels[i]->modified_trust.b;
        }

    return max_index;
}

void EBSL::init_base_weights()
{
    int model_count;
    base_weights.clear();
    float tmp, numerator, denominator = 0, weights_sum = 0;

    // Find the common denominator
    for (int i = 0; i < nb_models; ++i)
    {
        tmp = 1;
        for (int j = 0; j < nb_models; ++j)
        {
            if (j == i)
                continue;
            else
                tmp *= 1 - slmodels[j]->trust.b;
        }
        denominator += tmp;
    }
    for (int i = 0; i < nb_models; ++i)
    {
        numerator = slmodels[i]->trust.b;
        for (int j = 0; j < nb_models; ++j)
        {
            if (i == j)
                continue;
            else
                numerator *= 1 - slmodels[j]->trust.b;
        }
        base_weights.push_back(numerator / denominator);
        weights_sum += numerator / denominator;
    }

    // Take the weight of uncertainty into account
    base_weights.push_back(1 - weights_sum);
}
void EBSL::set_all_base_rates(float base_rate)
{
    for (int i = 0; i < nb_models; ++i)
        slmodels[i]->information.a = base_rate;
}

void EBSL::save_state(int64_t flow_id)
{
    try
    {
        auto state_entry = states_map.at(flow_id);
        // Update the stored state instead of allocating a new one with updated information
        state_entry.last_prediction_proba = last_prediction;
        state_entry.models_in_conflict = models_in_conflict;

        // Store updated bonuses and conflict counters
        for (int i = 0; i < nb_models; ++i)
        {
            state_entry.conflict_counters[i] = slmodels[i]->conflict_counter;
            state_entry.bonuses[i] = slmodels[i]->curr_bonus;
            state_entry.trust_offset[i] = slmodels[i]->trust_offset;
        }
        states_map[flow_id] = state_entry;
    }
    // Entry does not exist, create it
    catch (std::out_of_range e)
    {
        sl_state_snapshot new_entry;
        new_entry.last_prediction_proba = last_prediction;
        new_entry.models_in_conflict = models_in_conflict;
        // Store bonuses and conflict counters
        for (int i = 0; i < nb_models; ++i)
        {
            new_entry.conflict_counters.push_back(slmodels[i]->conflict_counter);
            new_entry.bonuses.push_back(slmodels[i]->curr_bonus);
            new_entry.trust_offset.push_back(slmodels[i]->trust_offset);
        }
        states_map[flow_id] = new_entry;
    }
}

void EBSL::load_state(int64_t flow_id)
{
    try
    {
        auto state_entry = states_map.at(flow_id);
        // Restore the last prediction
        last_prediction = state_entry.last_prediction_proba;
        models_in_conflict = state_entry.models_in_conflict;

        // Load updated bonuses and conflict counters and recalculate the modified trust
        for (int i = 0; i < nb_models; ++i)
        {
            slmodels[i]->conflict_counter = state_entry.conflict_counters[i];
            slmodels[i]->curr_bonus = state_entry.bonuses[i];
            slmodels[i]->trust_offset = state_entry.trust_offset[i];
            modify_trust(slmodels[i]->trust, slmodels[i]->trust_offset, slmodels[i]->modified_trust);
        }
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
    for (BSL_SM *slmodel : slmodels)
        slmodel->get_information_opinion(current_iteration);

    // Find model with the current highest trust and use its belief as the base rate (trust mode)
    if (base_rate_choice == TRUST_SOURCE)
    {
        int highest_trust_index = find_most_trusted();
        set_all_base_rates(slmodels[highest_trust_index]->information.b);
    }
    else if (base_rate_choice == PRIOR_SOURCE)
        set_all_base_rates(last_prediction);

    if (enable_debugging)
    {
        // Skipped above for optimization reasons (deferred until get_final_prediction) but needed here
        for (BSL_SM *slmodel : slmodels)
            slmodel->get_discounted_information_opinion();

        puts("* Initialization");
        puts("Information opinions:");
        for (BSL_SM *slmodel : slmodels)
        {
            printf("Model %s: ", slmodel->name.c_str());
            slmodel->information.print_opinion();
        }

        puts("\n* Before trust update");
        puts("Discounted information opinions:");
        for (BSL_SM *slmodel : slmodels)
        {
            printf("Model %s: ", slmodel->name.c_str());
            slmodel->discounted_information.print_opinion();
        }
    }
}

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
        float total_belief = 0;
        for (int i = 0; i < nb_models; ++i)
            total_belief += slmodels[i]->information.b;

        reference.b = total_belief / nb_models;
        reference.d = 1 - reference.b;
        reference.u = 0;
    }

    if (enable_debugging)
    {
        printf("Reference opinion: ");
        reference.print_opinion();
    }

    // Calculate conflict of each model with the reference
    // WARNING: It is assumed that both information and reference opinions are dogmatic
    // The original conflict equation is then reduced to simple belief distance
    for (BSL_SM *slmodel : slmodels)
        slmodel->conflict = fabs(slmodel->information.b - reference.b);
}

void EBSL::reevaluate_trust()
{
    float total_conflict = 0, average_conflict;
    for (BSL_SM *model : slmodels)
        total_conflict += model->conflict;
    average_conflict = total_conflict / nb_models;

    for (BSL_SM *slmodel : slmodels)
    {
        float distance_to_average_conf;
        distance_to_average_conf = slmodel->conflict - average_conflict;

        if (distance_to_average_conf > conflict_threshold)
        {
            // This is the first time this model is getting a penalty, increase the models in conflict counter
            if (slmodel->conflict_counter == 0)
                ++models_in_conflict;

            // Condition met, increment the conflict counter to increase penalty
            ++slmodel->conflict_counter;
            slmodel->trust_offset = get_penalty(slmodel->conflict_counter);

            // The model is predicting the positive class, so use the pclass bonus
            if (slmodel->information.b >= 0.5)
            {
                ++slmodel->pcumulative_conflict;
                // Used for bonus tuning (to know which models are worth giving bonuses)
                if (compare_to_true_labels && true_labels.data()[current_iteration] == true)
                    ++slmodel->pconflict_TP;

                slmodel->trust_offset -= slmodel->pclass_bonus;
                slmodel->curr_bonus = slmodel->pclass_bonus;
            }
            else
            {
                // Predicting the negative class
                ++slmodel->ncumulative_conflict;
                // Also for bonus tuning
                if (compare_to_true_labels && true_labels.data()[current_iteration] == false)
                    ++slmodel->nconflict_TN;

                slmodel->trust_offset -= slmodel->nclass_bonus;
                slmodel->curr_bonus = slmodel->nclass_bonus;
            }
            // Recalculate the modified trust opinion immediately (to avoid EBSL always recalculating all trust opinions later)
            modify_trust(slmodel->trust, slmodel->trust_offset, slmodel->modified_trust);
            // And the new discounted information opinion
            slmodel->get_discounted_information_opinion();
        }
        // Not enough conflict, start restoring trust if needed
        else if (slmodel->conflict_counter != 0)
        { // Gradually restoring trust
            slmodel->conflict_counter -= std::min(trust_restore_speed, slmodel->conflict_counter);

            if (slmodel->conflict_counter == 0)
            {
                // Reset both counters and decrement the "models in conflict" counter
                slmodel->trust_offset = slmodel->curr_bonus = 0;
                --models_in_conflict;
            }
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
        }

        // Store dist to average conflict, penalties and uncertainty
        for (int i = 0; i < nb_models; ++i)
        {
            slm_dist_to_avg[i].push_back(slmodels[i]->conflict - average_conflict);
            slm_penalties[i].push_back(slmodels[i]->trust_offset);
            slm_uncertainty[i].push_back(slmodels[i]->discounted_information.u);
        }
    }
}

float EBSL::get_final_prediction()
{
    // Reuse precalculated base weights to avoid the expensive average fusion
    if (models_in_conflict == 0)
    {
        float b = 0, d, u;
        for (int i = 0; i < nb_models; ++i)
            b += base_weights[i] * slmodels[i]->information.b;
        u = base_weights[nb_models];
        d = 1 - b - u;
        last_final_opinion.b = b;
        last_final_opinion.d = d;
        last_final_opinion.u = u;
    }
    else
    {
        // Before average fusion we must update the discounted information opinions
        for (BSL_SM *slmodel : slmodels)
            slmodel->get_discounted_information_opinion();
        update_discounted_info_vector();
        last_final_opinion = average_fusion(all_discounted_opinions);
    }

    // The base rate should be the same in every information opinion
    last_final_opinion.a = slmodels[0]->information.a;
    last_prediction = last_final_opinion.projected_probability();

    if (enable_debugging)
    {
        update_discounted_info_vector();
        puts("\n* Effective weights:");
        int most_trusted;
        if (base_rate_choice == TRUST_SOURCE)
            most_trusted = find_most_trusted();
        else
            most_trusted = -1;

        float denominator = 0;
        for (int i = 0; i < nb_models; ++i)
            denominator += uncertainty_product(all_discounted_opinions, i);

        // Now calculate the weight of each model
        for (int i = 0; i < nb_models; ++i)
        {
            float weight = slmodels[i]->modified_trust.b * uncertainty_product(all_discounted_opinions, i) / denominator;
            if (i == most_trusted)
                weight += last_final_opinion.u;
            printf("Model %s: %.3g\n", slmodels[i]->name.c_str(), weight);
            slm_weights[i].push_back(weight);
        }

        // Case of prior mode, the previous opinion has its weight
        if (base_rate_choice == PRIOR_SOURCE)
        {
            printf("Previous prediction: %.3g\n", last_final_opinion.u);
            slm_weights[nb_models].push_back(last_final_opinion.u);
        }

        printf("\n* Final opinion:");
        last_final_opinion.print_opinion();
        // Projected probability is made of 2 parts: belief and contribution of prior probability
        printf("Base rate contribution = %.3g\n", last_final_opinion.a * last_final_opinion.u);
        printf("Class 1 Probability = %.3g\n", last_prediction);

        slm_uncertainty[nb_models].push_back(last_final_opinion.u);
    }
    return last_prediction;
}

float EBSL::run_once()
{
    // The current flow ID, ignored if we are not in a multiflow run
    int64_t current_id;

    if (enable_debugging)
    {
        puts("--------------------------------------------");
        printf("Iteration %d\n", current_iteration);
        if (multi_flow)
            printf("Flow ID: %lli \n", id_list.data()[current_iteration]);
    }

    // Load the correct SL state for the current flow
    if (multi_flow)
    {
        current_id = id_list.data()[current_iteration];
        save_state(last_id);
        load_state(current_id);
    }

    get_all_information_opinions();
    get_ref_and_conflicts();
    // Based on conflicts, find new trust values
    reevaluate_trust();
    float class1_proba = get_final_prediction();

    // Update the last ID
    if (multi_flow)
        last_id = current_id;

    return class1_proba;
}

void EBSL::predict_proba()
{
    // Clear previous runs
    slm_dist_to_avg.clear();
    slm_weights.clear();
    slm_penalties.clear();
    slm_uncertainty.clear();
    models_in_conflict = 0;
    init_base_weights();

    // Preallocate debug vectors if needed
    if (enable_debugging)
    {
        slm_dist_to_avg.resize(nb_models);
        // Prior mode has the weight of the last prediction in addition to individual models
        slm_weights.resize(nb_models + (base_rate_choice == PRIOR_SOURCE ? 1 : 0));
        // +1 for the final opinion uncertainty
        slm_uncertainty.resize(nb_models + 1);
        slm_penalties.resize(nb_models);
    }

    // Clear the state map
    states_map.clear();

    // Clear old conflict statistics
    for (BSL_SM *m : slmodels)
    {
        m->pcumulative_conflict = m->pconflict_TP = 0;
        m->ncumulative_conflict = m->nconflict_TN = 0;
    }

    int nb_rows = class1_prediction.shape(0);

    float class1;
    auto results = class1_prediction.data();
    for (current_iteration = 0; current_iteration < nb_rows; ++current_iteration)
    {
        class1 = run_once();
        results[current_iteration] = class1;
    }
}

NB_MODULE(ebsl_cpp, m)
{
    nb::class_<EBSL>(m, "EBSL_cpp")
        .def(nb::init<>())
        .def(nb::init<float, float, float, float, int>(), "conflict_threshold"_a = 0.15, "max_penalty"_a = 0.5, "b"_a = 1., "trust_restore_speed"_a = 0.5, "base_rate_choice"_a = (int)PRIOR_SOURCE)
        .def("__str__", &EBSL::to_string)
        .def("add_model", &EBSL::add_model, "model"_a)
        .def("predict_proba", &EBSL::predict_proba)
        .def_rw("max_penalty", &EBSL::max_penalty)
        .def_rw("b", &EBSL::b)
        .def_rw("id_list", &EBSL::id_list)
        .def_rw("true_labels", &EBSL::true_labels)
        .def_rw("slmodels_dict", &EBSL::slmodels_map)
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
        .def_rw("iteration", &EBSL::current_iteration);

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
        .def_ro("pcumulative_conflict", &BSL_SM::pcumulative_conflict)
        .def_ro("pconflict_TP", &BSL_SM::pconflict_TP)
        .def_ro("ncumulative_conflict", &BSL_SM::ncumulative_conflict)
        .def_ro("nconflict_TN", &BSL_SM::nconflict_TN)
        .def_rw("name", &BSL_SM::name);

    m.def("average_fusion", &average_fusion, "all_opinions"_a);
    m.def("modify_trust", &modify_trust, "trust"_a, "offset"_a, "out"_a);
    m.def("uncertainty_product", &uncertainty_product, "all_opinions"_a, "exception_index"_a);
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
