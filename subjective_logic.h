#ifndef SL_INCLUDE
#define SL_INCLUDE

// Finds the length of stack allocated arrays
#define array_length(z) (sizeof(z) / sizeof(*z))

typedef struct opinion
{
    float b, d, u, a;
} opinion;

void print_opinion(opinion o);

// Checks if the belief is valid (b+d+u=1 and b,d,u,a in [0;1])
int validate_opinion(opinion o);

// Same as validate_opinion, but for an array of known length
int validate_opinions(opinion *all_opinions, int count);

// Calculates the averaging fusion of all opinions. Doesn't set the base rate
opinion average_fusion(opinion *all_opinions, int count);

// Discount the source opinion using the trust opinion
opinion trust_discounting(opinion trust, opinion source);

float projected_probability(opinion source);

// Find the conflict between the source opinion and a reference
float calculate_conflict(opinion reference, opinion source);
#endif