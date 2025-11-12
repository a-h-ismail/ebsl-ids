#ifndef BINOMIAL_OPINION_H
#define BINOMIAL_OPINION_H
#include <vector>
#include <string>
// The binomial opinion class
class Opinion
{
    friend class EBSL;
public:
    float b, d, u, a;

    Opinion();

    Opinion(float belief, float disbelief, float uncertainty, float base_rate);

    std::string to_string();

    void print_opinion();

    void set_parameters(float b, float d, float u, float a=-1);

    int validate_opinion();

    void trust_discounting(Opinion &trust, Opinion &out);

    float projected_probability();

    float calculate_conflict(Opinion reference);
};

// Check if belief opinions are valid (b+d+u=1 and b,d,u,a in [0;1])
int validate_opinions(Opinion *all_opinions, int count);

Opinion average_fusion(std::vector<Opinion> &all_opinions);

void modify_trust(Opinion &trust, float offset, Opinion &out);

// Helper function for belief fusion product of uncertainty with a single exception
float uncertainty_product(std::vector<Opinion> &all_opinions, int exception_index);

#endif