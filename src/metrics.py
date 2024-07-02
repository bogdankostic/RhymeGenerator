import string

import cmudict
import torch
from torchmetrics import Metric

from utils import pronunciations


phoneme_classes = {phone: ("vowel" if val == ['vowel'] else "consonant") for phone, val in cmudict.phones()}


class RhymeAccuracy(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: list[str], labels: list[list[str]]):
        for pred, label in zip(preds, labels):
            # Remove punctuation from pred and label
            pred = pred.translate(str.maketrans("", "", string.punctuation))
            label = label.translate(str.maketrans("", "", string.punctuation))

            # Get rhyme of pred and label
            pred_tokenized = pred.split()
            label_tokenized = label.split()
            pred_last_word = pred_tokenized[-1] if pred_tokenized else ""
            label_last_word = label_tokenized[-1] if label_tokenized else ""
            pred_rhymes = self._get_rhyme(pred_last_word)
            label_rhymes = self._get_rhyme(label_last_word)

            if any(pred_rhyme == label_rhyme for pred_rhyme in pred_rhymes for label_rhyme in label_rhymes):
                self.correct += 1
            self.total += 1

    def compute(self):
        return self.correct.float() / self.total

    @staticmethod
    def _get_rhyme(word: str):
        try:
            phonetic_transcriptions = pronunciations[word.lower().strip()]
            rhymes = []
            for variant in phonetic_transcriptions:
                vowel_seen = False
                rhyme = []
                for phoneme in reversed(variant):
                    if phoneme in phoneme_classes:
                        # We reached the onset -> we collected the rhyme
                        if vowel_seen and phoneme_classes[phoneme] == "consonant":
                            break

                        rhyme.insert(0, phoneme)
                        if phoneme_classes[phoneme] == "vowel":
                            vowel_seen = True

                rhymes.append(rhyme)
        except KeyError:
            rhymes = [[]]

        return rhymes
