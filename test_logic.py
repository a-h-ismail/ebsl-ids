#!/usr/bin/env python3
import subjective_logic as sl
import dummy_model as dm

a1 = sl.Opinion(0.8, 0.1, 0.1, 0.8)
a2 = sl.Opinion(0.9, 0.05, 0.05, 0.8)
a3 = sl.Opinion(0.1, 0.8, 0.1, 0.8)
a4 = sl.Opinion(0.5, 0.1, 0.4, 0.8)

fused = sl.average_fusion((a1, a2, a3, a4))
fused.set_base_rate(0.7)
print(fused)

print(a1.trust_discounting(a2))
print(a1.trust_discounting(a3))

for a in (a1, a2, a3, a4):
    print(a.calculate_conflict(fused))

print("------------")

m1 = dm.DModel((0.1, 0.1, 0.15, 0.6, 0.8, 0.6, 0.76, 0.81))
m2 = dm.DModel((0.05, 0.1, 0.8, 0.5, 0.9, 0.1, 0.81, 0.69))
m3 = dm.DModel((0.2, 0.1, 0.1, 0.2, 0.1, 0.4, 0.37, 0.2))

o1 = sl.Opinion(0.95, 0.02, 0.03)
o2 = sl.Opinion(0.8, 0.15, 0.05)
o3 = sl.Opinion(0.8, 0.1, 0.1)
eb1 = sl.BSL_SM(m1, o1)
eb2 = sl.BSL_SM(m2, o2)
eb3 = sl.BSL_SM(m3, o3)

# Create a new ensemble classifier and set the models
eclassifier = sl.EBSL(_debug=True)
eclassifier.add_model(eb1)
eclassifier.add_model(eb2)
eclassifier.add_model(eb3)

for i in range(8):
    print("Iteration", i)
    print("Prediction:", eclassifier.run_once([]))
    print("--------------------------------------------")
