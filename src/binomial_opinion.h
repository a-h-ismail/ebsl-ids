// Copyright (C) 2025 Ahmad Ismail
// SPDX-License-Identifier: MPL-2.0

/**
 * @file
 * @brief Defines the binomial opinion class and functions to operate on opinions
 */

#ifndef BINOMIAL_OPINION_H
#define BINOMIAL_OPINION_H
#include <vector>
#include <string>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

class Opinion
{
public:
    float b, d, u, a;

    Opinion();

    Opinion(float belief, float disbelief, float uncertainty, float base_rate = 0);

    std::string to_string();

    void print_opinion();

    void set_parameters(float b, float d, float u, float a = 0);

    void validate_opinion();

    void trust_discounting(Opinion &trust, Opinion &out);

    float projected_probability();

    float calculate_conflict(Opinion reference);
};

Opinion average_fusion(std::vector<Opinion> &all_opinions);

void modify_trust(Opinion trust, float offset, Opinion &out);

// Helper function for belief fusion product of uncertainty with a single exception
float uncertainty_product(std::vector<Opinion> &all_opinions, int exception_index);

#endif