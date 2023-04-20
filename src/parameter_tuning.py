import collections
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score


script_path = Path(__file__).resolve().parent

dataframe_groundtruth = pd.read_csv(f'{script_path}/../datasets/groundtruth.csv')
groundtruth = list(dataframe_groundtruth['has_under_extrusion'])[:]
for epoch in range(5, 20):
    dataframe_vote = pd.read_csv(f'{script_path}/../datasets/result_votes_{epoch}.csv')
    for threshold in range(1, 20):
        votes = list(dataframe_vote['has_under_extrusion'])[:]
        num_files = len(votes)
        final_pred = []
        for i in range(num_files):
            if votes[i] >= threshold:
                final_pred.append(1)
            else:
                final_pred.append(0)

        f1 = f1_score(final_pred, groundtruth, average='macro')
        f1_public = f1_score(final_pred[9679:], groundtruth[9679:], average='macro')
        f1_private = f1_score(final_pred[:9679], groundtruth[:9679], average='macro')
        print(f"Epoch: {epoch}, threshold: {threshold}, f1: {f1}, f1_public: {f1_public}, f1_private: {f1_private}")

