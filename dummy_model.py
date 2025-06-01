#!/usr/bin/env python3
import numpy as np


class DModel:
    "A very simple binary classifier model emulator. Predictions are generated according to a predefined sequence"

    def __init__(self, probability_seq=(), name="DModel") -> None:
        "The probability sequence is that of class 1"
        self.index = 0
        self.name = name
        self.set_sequence(probability_seq)

    def set_sequence(self, probability_seq: list | tuple):
        "The probability sequence is that of class 1"
        self.probability_seq = tuple(probability_seq)
        self.max_index = len(probability_seq)

        for prob in probability_seq:
            if prob < 0 or prob > 1:
                raise ValueError("All probabilitites should be in range [0;1]")

    def _increment_index(self):
        "Increments the index, automatically rolling over at the sequence end"
        if self.index == self.max_index:
            self.index = 0
        else:
            self.index += 1

    def _predict_one(self):
        # x is ignored intentionally
        prob = self.probability_seq[self.index]
        self._increment_index()
        return round(prob)

    def _predict_proba_one(self):
        # Provide the raw probability of both classes, intended to imitate sklearn
        probs = (1-self.probability_seq[self.index],
                 self.probability_seq[self.index])
        self._increment_index()
        return probs

    def predict(self, x) -> np.ndarray:
        nb_rows = len(x)
        results = np.empty(nb_rows)
        for i in range(nb_rows):
            results[i] = self._predict_one()
        return results

    def predict_proba(self, x) -> np.ndarray:
        nb_rows = len(x)
        results = np.empty((nb_rows, 2))

        for i in range(nb_rows):
            out = self._predict_proba_one()
            results[i][0] = out[0]
            results[i][1] = out[1]

        return results

    def reset(self):
        self.index = 0

    def __str__(self) -> str:
        return "Model:%s, values: %s" % (self.name, self.probability_seq)
