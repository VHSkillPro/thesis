import os
import itertools
import pandas as pd
import os.path as osp
from pathlib import Path
from random import shuffle
from sklearn.model_selection import train_test_split


def generate_pair_data(folder_path: str, test_size: float = 0.5):
    # Get identities
    identities = list(
        filter(
            lambda identity_name: osp.isdir(osp.join(folder_path, identity_name)),
            os.listdir(folder_path),
        )
    )

    # Generate genuine pairs
    genuine_pairs = []
    for identity_name in identities:
        images = os.listdir(osp.join(folder_path, identity_name))
        for img1, img2 in itertools.combinations(images, 2):
            image1_path = osp.join(folder_path, identity_name, img1)
            image2_path = osp.join(folder_path, identity_name, img2)
            genuine_pairs.append(image1_path, image2_path)

    imposter_pairs_left = [selfie1_path for (selfie1_path, _) in genuine_pairs]
    imposter_pairs_right = [selfie2_path for (_, selfie2_path) in genuine_pairs]

    # Generate imposter pairs
    def check():
        results = [
            Path(selfie2_path).parts[-2].split("_")[0] in selfie1_path
            for (selfie1_path, selfie2_path) in zip(
                imposter_pairs_left, imposter_pairs_right
            )
        ]
        return any(results)

    while check():
        shuffle(imposter_pairs_right)
    imposter_pairs = list(zip(imposter_pairs_left, imposter_pairs_right))

    # Split data
    genuine_pairs_train, genuine_pairs_test = train_test_split(
        genuine_pairs, test_size=test_size, random_state=42
    )
    imposter_pairs_train, imposter_pairs_test = train_test_split(
        imposter_pairs, test_size=test_size, random_state=42
    )

    # Save pairs data
    pairs_filepath = {
        "train": {
            "genuine_pairs": genuine_pairs_train,
            "imposter_pairs": imposter_pairs_train,
        },
        "test": {
            "genuine_pairs": genuine_pairs_test,
            "imposter_pairs": imposter_pairs_test,
        },
    }

    for phase in ["train", "test"]:
        for pair_type in ["genuine_pairs", "imposter_pairs"]:
            pairs_data = pairs_filepath[phase][pair_type]
            pairs_df = pd.DataFrame(pairs_data, columns=["image1_path", "image2_path"])
            pairs_df.to_csv(f"{pair_type}_{phase}.csv", index=False)


def load_pairs_data(folder_path: str, test_size: float = 0.5) -> dict:
    def check():
        return all(
            [
                osp.exists(osp.join(folder_path, f"{pair_type}_{phase}.csv"))
                for phase in ["train", "test"]
                for pair_type in ["genuine_pairs", "imposter_pairs"]
            ]
        )

    if not check():
        generate_pair_data(folder_path, test_size)

    pairs_data = {}
    for phase in ["train", "test"]:
        for pair_type in ["genuine_pairs", "imposter_pairs"]:
            file_path = osp.join(folder_path, f"{pair_type}_{phase}.csv")
            pairs_df = pd.read_csv(file_path)
            pairs_data[phase][pair_type] = pairs_df

    return pairs_data
