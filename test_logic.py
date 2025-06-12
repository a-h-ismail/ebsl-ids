#!/usr/bin/env python3
import subjective_logic as sl
import dummy_model as dm
import numpy as np

a1 = sl.Opinion(0.8, 0.1, 0.1, 0.8)
a2 = sl.Opinion(0.9, 0.05, 0.05, 0.8)
a3 = sl.Opinion(0.1, 0.8, 0.1, 0.8)
a4 = sl.Opinion(0.5, 0.1, 0.4, 0.8)

print("Basic example: conflict relative to the average reference:")
fused = sl.average_fusion((a1, a2, a3, a4))
fused.set_base_rate(0.7)
print("Average opinion:", fused)

i = 0
for a in (a1, a2, a3, a4):
    i += 1
    print("Conflict %d =" % i, a.calculate_conflict(fused))

print("-----------------------------------")
print("Test Subjective logic ensemble learning: EBSL")

# 5 dummy models to test the framework, they output these probabilities in order
# Assuming that true label is 0 0 0 1 1 1 1 1
# Assuming here that m1, m2 are the most trustworthy and m4,m5 struggling with this specific detection
# m3 is relatively uncertain in its decision
m1 = dm.DModel((0.1, 0.1, 0.15, 0.6, 0.8, 0.69, 0.76, 0.81))
m2 = dm.DModel((0.05, 0.1, 0.2, 0.5, 0.9, 0.7, 0.81, 0.4))
m3 = dm.DModel((0.2, 0.1, 0.1, 0.4, 0.3, 0.4, 0.37, 0.5))
m4 = dm.DModel((0.1, 0.13, 0.07, 0.3, 0.27, 0.4, 0.1, 0.21))
m5 = dm.DModel((0.1, 0.13, 0.07, 0.2, 0.13, 0.5, 0.32, 0.34))

o1 = sl.Opinion(0.95, 0.02, 0.03)
o2 = sl.Opinion(0.85, 0.1, 0.05)
o3 = sl.Opinion(0.9, 0.05, 0.05)
o4 = sl.Opinion(0.85, 0.1, 0.05)
o5 = sl.Opinion(0.8, 0.15, 0.05)
eb1 = sl.BSL_SM(m1, None, o1)
eb2 = sl.BSL_SM(m2, None, o2)
eb3 = sl.BSL_SM(m3, None, o3)
eb4 = sl.BSL_SM(m4, None, o4)
eb5 = sl.BSL_SM(m5, None, o5)

# Create a new ensemble classifier and set the models
eclassifier = sl.EBSL(_debug=True, base_rate_choice="prior",
                      trust_restore_speed=0.5,)
eclassifier.add_model(eb1)
eclassifier.add_model(eb2)
eclassifier.add_model(eb3)
eclassifier.add_model(eb4)
eclassifier.add_model(eb5)

print(eclassifier.predict(np.empty((8, 1))))
