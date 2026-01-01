#!/usr/bin/env python3

# Copyright (C) 2025 Ahmad Ismail
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import ebsl as sl
import dummy_model as dm
import numpy as np
from matplotlib import pyplot as plt

print("Test Subjective logic ensemble learning: EBSL")

# 3 dummy models to test the framework, they output these probabilities in order
# Assuming that true label is 0 0 0 1 1 1 1 1
# Assuming here that m1, is the most trustworthy and m2, m3 struggling with this specific detection
# You can add more samples as long as the length of true label and individual model probabilities are equal
true_label = (0, 0, 0, 1, 1, 1, 1, 1)

# Modify probabilities to observe behavior
m1 = dm.DModel((0.1, 0.15, 0.12, 0.2, 0.15, 0.77, 0.8, 0.69, 0.76, 0.81))
m2 = dm.DModel((0.05, 0.1, 0.07, 0.23, 0.2, 0.1, 0.3, 0.33, 0.41, 0.29))
m3 = dm.DModel((0.2, 0.1, 0.11, 0.14, 0.1, 0.22, 0.26, 0.4, 0.37, 0.5))


# Create a new ensemble classifier and set the models
eclassifier = sl.EBSL(_debug=True, base_rate_choice="prior", trust_restore_speed=0.5,
                      conflict_threshold=0.15, max_penalty=0.5, b=5)

eclassifier.add_model(sl.BSL_SM(m1, None, None, "1"))
eclassifier.add_model(sl.BSL_SM(m2, None, None, "2"))
eclassifier.add_model(sl.BSL_SM(m3, None, None, "3"))

eclassifier.get_model_by_name("1").trust_from_mcc(0.8)
eclassifier.get_model_by_name("2").trust_from_mcc(0.75)
eclassifier.get_model_by_name("3").trust_from_mcc(0.72)

# Comment/uncomment the next line to see the bonus effect on the first model
# eclassifier.get_model_by_name("1").set_bonuses(0, 0.2)

# As we are using dummy models, it doesn't need real samples, an empty array will do the trick
predicted = eclassifier.predict_proba(pd.DataFrame(np.empty((10, 1))))
print(eclassifier)
print(predicted)
print("EBSL:", predicted.round())
print("Hard Vote:", eclassifier._hard_vote())
print("Soft Vote:", eclassifier._soft_vote_prob())

# Plot the results
nb_samples = len(m1.probability_seq)
x = np.linspace(0, nb_samples-1, nb_samples)

# Create figure and axis objects with 2 rows and 2 columns
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.canvas.manager.set_window_title('EBSL Debug Statistics')  # type: ignore

axs[0, 0].set_title("Individual Model Class 1 Probabilities")
axs[0, 1].set_title("Distance to Average Conflict")
axs[0, 2].set_title("Weights")
axs[1, 0].set_title("Uncertainty in Information Opinion")
axs[1, 1].set_title("Penalty minus Bonus")
axs[1, 2].set_title("Final prediction probability")
axs[0, 0].set_ylim([0, 1])
axs[1, 2].set_ylim([0, 1])

# Plot debug metrics
for i in range(len(eclassifier._slmodels)):
    name = "Model %d" % (i+1)
    # Show probabilities
    axs[0, 0].plot(x, eclassifier._slmodels[i].prediction_cache, label=name, marker="o")
    # Dist to avg
    axs[0, 1].plot(x, eclassifier.slm_dist_to_avg[i], label=name, marker="o")
    # Weights
    axs[0, 2].plot(x, eclassifier.slm_weights[i], label=name, marker="o")
    # Uncertainty
    axs[1, 0].plot(x, eclassifier.slm_uncertainty[i], label=name, marker="o")
    # And modifiers
    axs[1, 1].plot(x, eclassifier.slm_penalties[i], label=name, marker="o")

# The conflict threshold line
axs[0, 1].axline((0, eclassifier.conflict_threshold), slope=0, color='r', label="Conflict Threshold")
# The last prediction weight
if eclassifier.base_rate_choice == "prior":
    axs[0, 2].plot(x, eclassifier.slm_weights[len(eclassifier._slmodels)], label="Last prediction", marker="x")
# And uncertainty
axs[1, 0].plot(x, eclassifier.slm_uncertainty[len(eclassifier._slmodels)], label="Final opinion", marker="o")

# Prediction comparison between voting classifiers
axs[1, 2].plot(x, predicted, label="EBSL", marker='x')
axs[1, 2].plot(x, eclassifier._hard_vote(), label="Hard Vote", marker='x')
axs[1, 2].plot(x, eclassifier._soft_vote_prob(), label="Soft Vote", marker='x')
axs[1, 2].axline((0, 0.5), slope=0, color='r', label="Probability Threshold")

# Write the legends
for i in range(2):
    for j in range(3):
        box = axs[i, j].get_position()
        axs[i, j].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        axs[i, j].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.show()
