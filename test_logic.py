#!/usr/bin/env python3
import pandas as pd
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

# 3 dummy models to test the framework, they output these probabilities in order
# Assuming that true label is 0 0 0 1 1 1 1 1
# Assuming here that m1, is the most trustworthy and m2, m3 struggling with this specific detection
m1 = dm.DModel((0.1, 0.1, 0.15, 0.77, 0.8, 0.69, 0.76, 0.81))
m2 = dm.DModel((0.05, 0.1, 0.2, 0.1, 0.3, 0.33, 0.81, 0.6))
m3 = dm.DModel((0.2, 0.1, 0.1, 0.22, 0.26, 0.4, 0.37, 0.5))

o1 = sl.Opinion(0.95, 0.02, 0.03)
o2 = sl.Opinion(0.85, 0.1, 0.05)
o3 = sl.Opinion(0.8, 0.15, 0.05)

eb1 = sl.BSL_SM(m1, None, trust_opinion=o1)
eb2 = sl.BSL_SM(m2, None, trust_opinion=o2)
eb3 = sl.BSL_SM(m3, None, trust_opinion=o3)

# Create a new ensemble classifier and set the models
eclassifier = sl.EBSL(_debug=True, base_rate_choice="prior", trust_restore_speed=0.5,
                      conflict_threshold=0.2, max_penalty=0.7)
eclassifier.add_model(eb1)
eclassifier.add_model(eb2)
eclassifier.add_model(eb3)

# Comment/uncomment the next line to see the bonus effect
eb1.set_bonuses(0, 0.7)

print(eclassifier.predict(pd.DataFrame(np.empty((8, 1)))))
