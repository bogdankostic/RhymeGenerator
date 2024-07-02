import cmudict
import pandas as pd
import re


def get_verse_pairs(df: pd.DataFrame, use_reverse: bool = True):
    if not use_reverse:
        df = df[df["reversed"] == False]

    verse_a = df["verse_a"].tolist()
    verse_b = df["verse_b"].tolist()

    return {"verse_a": verse_a, "verse_b": verse_b}


def cmu_pronunciations_without_stress():
    phonetic_repr = cmudict.dict()
    phonetic_repr_without_stress = {}
    for word, pronunciation_vars in phonetic_repr.items():
        vars_without_stress = []
        for pronunciation in pronunciation_vars:
            vars_without_stress.append([re.sub(r"\d", "", phoneme) for phoneme in pronunciation])

        phonetic_repr_without_stress[word] = vars_without_stress

    return phonetic_repr_without_stress


pronunciations = cmu_pronunciations_without_stress()
