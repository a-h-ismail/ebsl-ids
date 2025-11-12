#include <iostream>
#include <math.h>
#include "binomial_opinion.h"
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

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

void Opinion::set_parameters(float b, float d, float u, float a)
{
    Opinion::b = b;
    Opinion::d = d;
    Opinion::u = u;
    Opinion::a = a;
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
}

std::string Opinion::to_string()
{
    char tmp[100];
    sprintf(tmp,"belief = %g, disbelief = %g, uncertainty = %g, base rate = %g", b, d, u, a);
    return std::string(tmp);
}

// The binomial opinion class
void Opinion::print_opinion()
{
    printf("belief = %g, disbelief = %g, uncertainty = %g, base rate = %g", b, d, u, a);
}

int Opinion::validate_opinion()
{
    if (valid_proba(b) != 0 || valid_proba(d != 0) || valid_proba(u != 0))
        return 1;
    if (fabs(1 - b - d - u) > 1e-5)
        return 1;
    else
        return 0;
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
    if (fused.validate_opinion() != 0)
        exit(1);

    return fused;
}

void Opinion::trust_discounting(Opinion &trust, Opinion &out)
{
    out.b = trust.b * b;
    out.d = trust.b * d;
    out.u = 1 - out.b - out.d;
    out.a = a;
}

void modify_trust(Opinion &trust, float offset, Opinion &out)
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

// Check if belief opinions are valid (b+d+u=1 and b,d,u,a in [0;1])
int validate_opinions(Opinion *all_opinions, int count)
{
    for (int i = 0; i < count; ++i)
        if (all_opinions[i].validate_opinion() != 0)
            return 1;

    return 0;
}

NB_MODULE(binomial_opinion, m)
{
    m.def("average_fusion", &average_fusion);
    m.def("modify_trust", &modify_trust);
    m.def("uncertainty_product", &uncertainty_product);
    nb::class_<Opinion>(m, "Opinion")
        .def(nb::init<>())
        .def(nb::init<float,float,float,float>(), "b"_a, "d"_a, "u"_a,"a"_a=1)
        .def("__str__",&Opinion::to_string)
        .def("validate_opinions", &Opinion::validate_opinion)
        .def("set_parameters", &Opinion::set_parameters,"b"_a, "d"_a, "u"_a,"a"_a=1)
        .def("calculate_conflict", &Opinion::calculate_conflict,"reference"_a)
        .def("trust_discounting", &Opinion::trust_discounting,"trust"_a,"out"_a)
        .def("projected_probability", &Opinion::projected_probability)
        .def("print_opinion", &Opinion::print_opinion)
        .def_rw("b", &Opinion::b)
        .def_rw("d", &Opinion::d)
        .def_rw("u", &Opinion::u)
        .def_rw("a", &Opinion::a);
}
