#ifndef EBSL_CPP
#define EBSL_CPP

#include "binomial_opinion.h"
#include "bsl_sm.h"
#include <unordered_map>
#include <stdexcept>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <cstdio>
enum
{
    PRIOR_SOURCE,
    TRUST_SOURCE
};

typedef struct sl_state_snapshot
{
    float last_prediction_proba;
    std::vector<float> conflict_counters, bonuses, trust_offset;
} sl_state_snapshot;

namespace nb = nanobind;
using namespace nb::literals;

class EBSL
{
    // EBSL: Ensemble Binomial Subjective Logic
    /*
        Collection of BSL_SM *models. Enables prediction aggregation using subjective logic

        Parameters
        ----------
        conflict_threshold: The minimum value a model's conflict should have above the average conflict to reduce its trust

        max_penalty: The maximun penalty added to the model's trust opinion (disbelief)

        b: Inverse of speed of losing trust

        trust_restore_speed: Indicates trust restoration step size on lack of conflict (mistrust step size is 1)

        base_rate_choice: How to choose the base rate. "prior" uses the last aggregated prediction probability.
        "trust" uses the probability produced by the currently most trusted model

        _debug: Enables debugging output
        */

private:
    Opinion reference, last_final_opinion, empty_opinion;
    float last_prediction;
    int64_t last_id;
    std::unordered_map<int64_t, sl_state_snapshot> state_store;
    std::vector<Opinion> all_discounted_opinions;
    std::vector<float> all_conflicts;

public:
    std::vector<BSL_SM *> slmodels;
    float conflict_threshold, max_penalty, b, trust_restore_speed;
    bool enable_debugging, compare_to_true_labels;
    int base_rate_choice;
    nb::ndarray<bool, nb::numpy, nb::shape<-1>, nb::c_contig> true_labels;
    nb::ndarray<int64_t, nb::numpy, nb::shape<-1>, nb::c_contig> id_list;
    nb::ndarray<float, nb::numpy, nb::shape<-1>, nb::c_contig> class1_prediction;
    bool multi_flow;
    int iteration, iterations_count;

    // For debugging and statistics collection
    std::vector<std::vector<float>> slm_dist_to_avg, slm_uncertainty, slm_penalties, slm_weights;

    EBSL(float conflict_threshold = 0.15, float max_penalty = 0.5, float b = 1., float trust_restore_speed = 0.5, int base_rate_choice = PRIOR_SOURCE);

    void clear_sl_state();

    std::string to_string();

    void add_model(BSL_SM *model);

    // def get_model_by_name(self, name: str) -> BSL_SM:
    //     return _slmodels_dict[name]

    // Returns the index of the highest belief in modified trust opinions
    int find_most_trusted();

    void set_all_base_rates(float base_rate);

    void save_state(int64_t flow_id);

    void load_state(int64_t flow_id);

    float get_penalty(float nb_conflict);

    void get_all_information_opinions();

    // Calculate conflict relative to the reference opinion. Results are stored in each model object
    void get_ref_and_conflicts();

    // Updates the trust for each model according to the conflict
    void reevaluate_trust();

    // Calculates the final prediction using discounted information opinion
    float get_final_prediction();

    float run_once();

    // Run the ebsl algorithm on already stored predictions
    // If true_labels is also provided, collects CICR statistics
    void predict_proba();
};

#endif