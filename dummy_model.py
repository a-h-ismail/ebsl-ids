#!/usr/bin/env python3

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

    def predict(self, x):
        # x is ignored intentionally
        prob = self.probability_seq[self.index]
        self.index += 1
        return round(prob)

    def predict_proba(self, x):
        # Provide the raw probability of both classes, intended to imitate sklearn
        probs = (1-self.probability_seq[self.index],
                 self.probability_seq[self.index])
        self.index += 1
        return probs

    def reset(self):
        self.index = 0

    def __str__(self) -> str:
        return "Model:%s, values: %s" % (self.name, self.probability_seq)
