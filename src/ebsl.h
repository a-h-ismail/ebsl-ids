// Copyright (C) 2025-2026 Ahmad Ismail
// SPDX-License-Identifier: MPL-2.0

/**
 * @file
 * @brief Declares the EBSL class and supporting structures
 */

#ifndef EBSL_CPP
#define EBSL_CPP

#include "binomial_opinion.h"
#include "bsl_sm.h"
#include <mutex>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>
#include <unordered_map>

enum base_rate_sourcing_mode
{
    PRIOR_SOURCE,
    TRUST_SOURCE
};

/// Subjective logic state store, for efficient save/load
typedef struct sl_state_snapshot
{
    float last_prediction_proba;
    int models_in_conflict;
    std::vector<float> conflict_counters, bonuses, trust_offset;
} sl_state_snapshot;

namespace nb = nanobind;
using namespace nb::literals;

/// The Ensemble Binomial Subjective Logic class, manages BSL_SM instances and implements the ensembling algorithm.
/// @note Python Bindings expose this class as EBSL_cpp
class EBSL
{
private:
    Opinion reference, last_final_opinion;
    float last_prediction;
    int64_t last_id;
    std::unordered_map<int64_t, sl_state_snapshot> states_map;
    std::vector<Opinion> all_discounted_opinions;
    std::vector<float> base_weights;
    int nb_models;
    int models_in_conflict;
    int64_t current_iteration;
    int64_t iterations_count;
    std::mutex lock;

    /// Initializes base weights, which is the weight of every model when no trust modification is occuring. Helps speed up the case of zero trust offset.
    void init_base_weights();

    /// Clears the subjective logic state of the EBSL and BSL_SM instances
    void clear_sl_state();

    /// Sets base_rate (a) to all original information opinions
    void set_all_base_rates(float base_rate);

    /// Returns the index of the highest belief in modified trust opinions
    int find_most_trusted();

    /// Updates the vector "all_discounted_opinions"
    void update_discounted_info_vector();

    /// Saves the current subjective logic state to the state of flow_id
    void save_state(int64_t flow_id);

    /// Loads the subjective logic state of flow_id
    void load_state(int64_t flow_id);

    /// Gets all original model opinions (from the state cache) then calculates the reference opinion if necessary
    void get_all_information_opinions();

    /// Calculate conflict relative to the reference opinion. Results are stored in each model object
    void get_ref_and_conflicts();

    /// Updates the trust for each model according to the conflict
    void reevaluate_trust();

    /// Calculates the final prediction using discounted information opinions
    float get_final_prediction();

    /// Executes one iteration of the EBSL algorithm. Does not modify the current_iteration counter
    float run_once();

    void _prepare_predictor();

public:
    /**
     * Contains pointers to all instances of BSL_SM objects.
     * @warning You are responsible for freeing the memory ocuppied by these objects if necessary.
     */
    std::vector<BSL_SM *> slmodels;

    /// Same as slmodels, but indexable by the model's name.
    std::unordered_map<std::string, BSL_SM *> slmodels_map;

    /// When a BSL_SM gets conflict higher than this threshold, it will get a penalty. Otherwise trust is gradually restored.
    float conflict_threshold;

    /// The highest penalty value possible
    float max_penalty;

    /// Parameter for the penalty function
    float b;

    /// The step size when reducing the penalty counter
    float trust_restore_speed;

    /// Enables/disables debugging output
    bool enable_debugging;

    /// When enabled, the EBSL collects CICR related statistics for each model (pcumulative_conflict ,pconflict_TP, ...)
    bool compare_to_true_labels;

    /// Enable/disable multi-flow support
    bool multi_flow;

    /// Base rate sourcing mode, see enum "base_rate_sourcing_mode"
    int base_rate_choice;

    /// List of true labels, mandatory when compare_to_true_labels is set to true
    nb::ndarray<bool, nb::numpy, nb::shape<-1>, nb::c_contig> true_labels;

    /// List of flow IDs, for multi-flow tracking
    nb::ndarray<int64_t, nb::numpy, nb::shape<-1>, nb::c_contig> id_list;

    /// For debugging and statistics collection
    std::vector<std::vector<float>> slm_dist_to_avg, slm_uncertainty, slm_penalties, slm_weights;

    EBSL(float conflict_threshold = 0.15, float max_penalty = 0.5, float b = 1., float trust_restore_speed = 0.5, int base_rate_choice = PRIOR_SOURCE);

    std::string to_string();

    /// Adds a model to the ensemble
    void add_model(BSL_SM *model);

    /// Removes the model with the specified name
    void remove_model(std::string name);

    /// Clears all BSL_SM models from the classifier
    void clear_all_models();

    /// Returns the penalty corresponding to the provided nb_conflict counter
    float get_penalty(float nb_conflict);

    /// Run the EBSL algorithm on already stored predictions
    /// @note If true_labels is also provided, collects CICR statistics
    void predict_proba(nb::ndarray<float, nb::numpy, nb::shape<-1>, nb::c_contig> out);

    void predict(nb::ndarray<uint8_t, nb::numpy, nb::shape<-1>, nb::c_contig> out);
};

#endif