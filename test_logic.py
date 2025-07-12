#!/usr/bin/env python3
import pandas as pd
import subjective_logic as sl
import dummy_model as dm
import numpy as np

print("Test Subjective logic ensemble learning: EBSL")

# 3 dummy models to test the framework, they output these probabilities in order
# Assuming that true label is 0 0 0 1 1 1 1 1
# Assuming here that m1, is the most trustworthy and m2, m3 struggling with this specific detection
true_label = (0, 0, 0, 1, 1, 1, 1, 1)
m1 = dm.DModel((0.1, 0.1, 0.15, 0.77, 0.8, 0.69, 0.76, 0.81))
m2 = dm.DModel((0.05, 0.1, 0.2, 0.1, 0.3, 0.33, 0.81, 0.29))
m3 = dm.DModel((0.2, 0.1, 0.1, 0.22, 0.26, 0.4, 0.37, 0.5))

o1 = sl.Opinion(0.95, 0.02, 0.03)
o2 = sl.Opinion(0.85, 0.1, 0.05)
o3 = sl.Opinion(0.8, 0.15, 0.05)

eb1 = sl.BSL_SM(m1, None, o1, "1")
eb2 = sl.BSL_SM(m2, None, o2, "2")
eb3 = sl.BSL_SM(m3, None, o3, "3")

# Create a new ensemble classifier and set the models
eclassifier = sl.EBSL(_debug=True, base_rate_choice="trust", trust_restore_speed=0.5,
                      conflict_threshold=0.2, max_penalty=0.7)
eclassifier.add_model(eb1)
eclassifier.add_model(eb2)
eclassifier.add_model(eb3)

# Comment/uncomment the next line to see the bonus effect
eb1.set_bonuses(0, 0.7)

# As we are using dummy models, it doesn't need real samples, an empty array will do the trick
predicted = eclassifier._predict_proba(pd.DataFrame(np.empty((8, 1))))
print(eclassifier)
print(predicted)
print("EBSL:", predicted.round())
print("Hard Vote:", eclassifier._hard_vote())
print("Soft Vote:", eclassifier._soft_vote_prob())
