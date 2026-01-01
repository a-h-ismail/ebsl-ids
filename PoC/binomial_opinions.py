#!/usr/bin/env python3

# Copyright (C) 2025 Ahmad Ismail
# SPDX-License-Identifier: MPL-2.0

from ebsl import Opinion, average_fusion

# Create new binomial opinions
o1 = Opinion(0.7, 0.2, 0.1, 0.6)
o2 = Opinion(0.4, 0.2, 0.4, 0.6)
o3 = Opinion(0.95, 0.03, 0.02, 0.6)

# Demonstrate probability projection
i = 1
for oi in (o1, o2, o3):
    print("Opinion %d (%s) projected probability: %g" % (i, oi, oi.projected_probability()))
    i += 1
print()

# Trust discounting section
high_trust = Opinion(0.9, 0.05, 0.05)
low_trust = Opinion(0.2, 0.7, 0.1)

out = Opinion()
o3.trust_discounting(high_trust, out)
print("Trust discounting of o3 with high trust (high belief):", out)
o3.trust_discounting(low_trust, out)
print("Trust discounting of o3 with low trust (low belief):", out)

# Average fusion example
print("\nAverage of o1, o2, o3:", average_fusion((o1, o2, o3)))
