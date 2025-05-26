#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct opinion
{
    float b, d, u, a;
} opinion;

void print_opinion(opinion o)
{
    printf("belief = %g, disbelief = %g, uncertainty = %g, base rate = %g\n", o.b, o.d, o.u, o.a);
}

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

int validate_opinion(opinion o)
{
    if (valid_proba(o.b) != 0 || valid_proba(o.d != 0) || valid_proba(o.u != 0))
        return 1;
    if (fabs(1 - o.b - o.d - o.u) > 1e-5)
        return 1;
    else
        return 0;
}

// Check if belief opinions are valid (b+d+u=1 and b,d,u,a in [0;1])
int validate_opinions(opinion *all_opinions, int count)
{
    for (int i = 0; i < count; ++i)
        if (validate_opinion(all_opinions[i]) != 0)
            return 1;

    return 0;
}

// Helper function for belief fusion product of uncertainty with a single exception
float _uncertainty_product(opinion *all_opinions, int count, int exception_index)
{
    float product = 1;
    for (int i = 0; i < count; ++i)
    {
        if (i == exception_index)
            continue;
        else
            product *= all_opinions[i].u;
    }
    return product;
}

opinion average_fusion(opinion *all_opinions, int count)
{
    opinion fused = {0, 0, 0, 0};
    float numerator = 0, denominator = 0, u_product;
    // Calculating bx
    for (int i = 0; i < count; ++i)
    {
        u_product = _uncertainty_product(all_opinions, count, i);
        numerator += all_opinions[i].b * u_product;
        denominator += u_product;
    }
    fused.b = numerator / denominator;

    // Calculating ux
    denominator = 0;
    for (int i = 0; i < count; ++i)
    {
        u_product = _uncertainty_product(all_opinions, count, i);
        denominator += u_product;
    }
    u_product = _uncertainty_product(all_opinions, count, -1);
    numerator = u_product;
    fused.u = numerator * count / denominator;

    fused.d = 1 - fused.b - fused.u;
    if (validate_opinion(fused) != 0)
        exit(1);

    return fused;
}

opinion trust_discounting(opinion trust, opinion source)
{
    opinion discounted;
    discounted.b = trust.b * source.b;
    discounted.d = trust.b * source.d;
    discounted.u = trust.d + trust.u + trust.b * source.u;
    discounted.a = source.a;
    return discounted;
}

float projected_probability(opinion source)
{
    return source.b + source.a * source.u;
}

float calculate_conflict(opinion reference, opinion source)
{
    float pd, cc;
    pd = fabs(projected_probability(reference) - projected_probability(source));
    cc = (1 - reference.u) * (1 - source.u);
    return pd * cc;
}

int main()
{
    // This section may be later moved elsewhere if converting this source file to a library is needed
    // Kept here for simple testing

    // Assuming the correct prediction is 0: no attack
    opinion a1 = {0.2, 0.7, 0.1, 0.8}, a2 = {0.799, 0.2, 0.001, 0.8}, a3 = {0.25, 0.65, 0.1, 0.8}, a4 = {0.1, 0.8, 0.1, 0.8};
    // Trust opinions
    opinion t_high = {0.85, 0.05, 0.1, 0.8}, t_low = {0.05, 0.85, 0.1, 0.8}, tmp;
    opinion all_opinions[] = {a1, a2, a3, a4};
    opinion ref;

    if (validate_opinions(all_opinions, array_length(all_opinions)) != 0)
        return 1;

    puts("Average fusion:");
    for (int i = 0; i < 1e7; ++i)
        ref = average_fusion(all_opinions, array_length(all_opinions));

    print_opinion(ref);
    putchar('\n');

    tmp = trust_discounting(t_high, a1);
    puts("a1: Before discounting:");
    print_opinion(a1);
    puts("After discounting (high trust):");
    print_opinion(tmp);
    tmp = trust_discounting(t_low, a1);
    puts("After discounting (low trust):");
    print_opinion(tmp);

    puts("\nConflict (before fusion)");
    float conflict;
    for (int i = 0; i < array_length(all_opinions); ++i)
    {
        conflict = calculate_conflict(ref, all_opinions[i]);
        printf("a%d conflict= %g\n", i + 1, conflict);
    }

    return 0;
}
