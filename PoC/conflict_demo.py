#!/usr/bin/env python3

# Copyright (C) 2026 Ahmad Ismail
# SPDX-License-Identifier: MPL-2.0

from ebsl import Opinion

wref = Opinion(0.67, 0.24, 0.09, 0.8)

w1 = Opinion(0.8, 0.1, 0.1, 0.8)
w2 = Opinion(0.9, 0.05, 0.05, 0.8)
w3 = Opinion(0.1, 0.8, 0.1, 0.8)
w4 = Opinion(0.5, 0.1, 0.4, 0.8)
w5 = Opinion(0.1, 0.5, 0.4, 0.8)

all_opinions = (w1, w2, w3, w4, w5)

for i in range(5):
    print("Conflict of w%d relative to w_ref = %.3g" % (i+1, all_opinions[i].calculate_conflict(wref)))
