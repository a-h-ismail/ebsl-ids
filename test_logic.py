#!/usr/bin/env python3
import subjective_logic as sl

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
