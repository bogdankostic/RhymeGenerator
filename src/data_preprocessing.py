import glob
import os
from collections import defaultdict
from itertools import combinations
import re

import pandas as pd
from sklearn.model_selection import train_test_split


def extract_rhyme_pairs(directory: str, output_file: str):
    """
    Extracts all rhyme pairs from poems in the given directory and writes them as csv to the output file.

    :param directory: The directory containing the poems grouped by author.
    :param output_file: The file to write the rhyme pairs to.
    """
    verse_a = []
    verse_b = []
    author = []
    title = []
    reversed_ = []

    for file in glob.glob(f"{directory}/*.txt"):
        with open(file, "r", encoding="cp1252", errors="ignore") as f:
            content = f.read()

        # Extract the author of the poems
        cur_author = re.search(r'AUTHOR\s(.+)', content).group(1).strip()

        # Split the file content by TITLE section to process each poem separately
        poems = re.split(r"TITLE", content)[1:]
        for poem in poems:
            title_match = re.match(r'(.+)\n', poem)
            if title_match:
                cur_title = title_match.group(1).strip()
                poem = poem[title_match.end():]
            else:
                cur_title = ""

            # Split the poem into sections containing different rhyme schemes
            sections = re.split(r'(RHYME\s+[a-z ]+\*?)', poem)
            # Extract rhyme pairs from each section
            for idx in range(1, len(sections), 2):
                rhyme_pattern = sections[idx].strip()[len("RHYME "):]
                verses = [verse.strip() for verse in sections[idx+1].strip().split("\n")]

                # Rhyme pattern is repeated
                if rhyme_pattern.endswith("*"):
                    rhyme_pattern = rhyme_pattern[:-1]

                rhyme_pattern = rhyme_pattern.split()
                pattern_len = len(rhyme_pattern)
                pattern_repeats = len(verses) // len(rhyme_pattern)
                rhyme_pattern = rhyme_pattern * pattern_repeats

                # Extract all verses that rhyme with each other
                rhyming_verses = defaultdict(list)
                for index, (verse, group) in enumerate(zip(verses, rhyme_pattern)):
                    group_key = f"{group}_{index // pattern_len}"
                    rhyming_verses[group_key].append(verse)

                for cur_rhyming_group in rhyming_verses.values():
                    if len(cur_rhyming_group) >= 2:
                        cur_pairs = combinations(cur_rhyming_group, 2)
                        for cur_verse_a, cur_verse_b in cur_pairs:
                            verse_a.append(cur_verse_a)
                            verse_b.append(cur_verse_b)
                            author.append(cur_author)
                            title.append(cur_title)
                            reversed_.append(False)

                            verse_a.append(cur_verse_b)
                            verse_b.append(cur_verse_a)
                            author.append(cur_author)
                            title.append(cur_title)
                            reversed_.append(True)


    df = pd.DataFrame({"verse_a": verse_a, "verse_b": verse_b, "author": author, "title": title, "reversed": reversed_})

    # Deduplicate the dataframe
    df = df.drop_duplicates()

    # Save as csv
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_file, index=False)


def train_test_dev_split(input_file: str, output_dir: str, train_size: float = 0.6, test_size: float = 0.2, random_state: int = 42):
    """
    Splits the rhyme pairs in the input file into train, test and dev sets and writes them to separate files.

    :param input_file: The file containing the rhyme pairs.
    :param output_dir: The directory to write the train, test and dev files to.
    :param train_size: The proportion of the data to include in the train set.
    :param test_size: The proportion of the data to include in the test set.
    :param random_state: The seed used to randomly split the data.
    """
    df = pd.read_csv(input_file)
    train, test_dev = train_test_split(df, train_size=train_size, random_state=random_state)
    test, dev = train_test_split(test_dev, train_size=1 - (test_size / (1 - train_size)), random_state=random_state)

    train_output = os.path.join(output_dir, "train.csv")
    test_output = os.path.join(output_dir, "test.csv")
    dev_output = os.path.join(output_dir, "dev.csv")

    train.to_csv(train_output, index=False)
    test.to_csv(test_output, index=False)
    dev.to_csv(dev_output, index=False)


if __name__ == "__main__":
    extract_rhyme_pairs("../data/raw", "../data/processed/rhyme_pairs.csv")
    train_test_dev_split("../data/processed/rhyme_pairs.csv", "../data/processed")
