// Copyright (C) 2025 Ahmad Ismail
// SPDX-License-Identifier: MPL-2.0

#include <iostream>
#include <math.h>
#include "binomial_opinion.h"
#include <format>

namespace nb = nanobind;
using namespace nb::literals;

// Some helper functions
int value_between(float value, float lowerbound, float upperbound)
{
    if (value < lowerbound)
        return -1;
    if (value > upperbound)
        return 1;
    else
        return 0;
}

int valid_proba(float value)
{
    return value_between(value, 0, 1);
}

void Opinion::set_parameters(float belief, float disbelief, float uncertainty, float base_rate)
{
    b = belief;
    d = disbelief;
    u = uncertainty;
    a = base_rate;
    validate_opinion();
}

Opinion::Opinion()
{
    b = d = u = a = 0;
}

Opinion::Opinion(float belief, float disbelief, float uncertainty, float base_rate)
{
    b = belief;
    d = disbelief;
    u = uncertainty;
    a = base_rate;
    validate_opinion();
}

std::string Opinion::to_string()
{
    return std::format("b = {:g}, d = {:g}, u = {:g}, a = {:g}", b, d, u, a);
}

// The binomial opinion class
void Opinion::print_opinion()
{
    printf("b = %g, d = %g, u = %g, a = %g\n", b, d, u, a);
}

void Opinion::validate_opinion()
{
    if (valid_proba(b) != 0 || valid_proba(d) != 0 || valid_proba(u) != 0 || valid_proba(a) || fabs(1 - b - d - u) > 1e-6)
        throw std::range_error("b, d, u, a must be in range [0;1] and b+d+u=1");
}

// Helper function for belief fusion product of uncertainty with a single exception
float uncertainty_product(std::vector<Opinion> &all_opinions, int exception_index)
{
    float product = 1;
    for (int i = 0; i < all_opinions.size(); ++i)
    {
        if (i == exception_index)
            continue;
        else
            product *= all_opinions[i].u;
    }
    return product;
}

Opinion average_fusion(std::vector<Opinion> &all_opinions)
{
    Opinion fused;
    int count = all_opinions.size();
    float numerator = 0, denominator = 0, u_product;
    // Calculating bx
    for (int i = 0; i < count; ++i)
    {
        u_product = uncertainty_product(ref(all_opinions), i);
        numerator += all_opinions[i].b * u_product;
        denominator += u_product;
    }
    fused.b = numerator / denominator;

    // Calculating ux
    denominator = 0;
    for (int i = 0; i < count; ++i)
    {
        u_product = uncertainty_product(all_opinions, i);
        denominator += u_product;
    }
    u_product = uncertainty_product(all_opinions, -1);
    numerator = u_product;
    fused.u = numerator * count / denominator;

    fused.d = 1 - fused.b - fused.u;
    // Should never fail but keep it just in case
    fused.validate_opinion();

    return fused;
}

void Opinion::trust_discounting(Opinion &trust, Opinion &out)
{
    out.b = trust.b * b;
    out.d = trust.b * d;
    out.u = 1 - out.b - out.d;
    out.a = a;
}

void modify_trust(Opinion trust, float offset, Opinion &out)
{
    // -d <= offset <= b
    // Reminder: b + d + u = 1 and b,d >= 0
    if (offset > trust.b)
        offset = trust.b;
    else if (offset < -trust.d)
        offset = -trust.d;
    // If d and u becomes 0, u in discounted opinions may become zero, causing problems
    if (trust.u == 0)
        offset += 0.05;

    out.b = trust.b - offset;
    out.d = trust.d + offset;
}

float Opinion::projected_probability()
{
    return b + a * u;
}

float Opinion::calculate_conflict(Opinion reference)
{
    float pd, cc;
    pd = fabs(reference.projected_probability() - projected_probability());
    cc = (1 - reference.u) * (1 - u);
    return pd * cc;
}
