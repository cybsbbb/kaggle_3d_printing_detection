import collections
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

ground_truth = {
                # Private Result
                '101-1678578332': 1, '101-1678343246': 0,
                '102-1678764144': 1, '102-1678744931': 0,
                '23-1674179039': 1, '23-1674179701': 1, '23-1674180283': 1, '23-1674180772': 1, '23-1674181142': 1,
                '23-1674182025': 1, '23-1674182884': 1, '23-1674183481': 1, '23-1674185002': 1, '23-1674185462': 1,
                '23-1674184477': 1, '23-1674186076': 1, '23-1674187131': 1, '23-1674183959': 1, '23-1674184223': 1,
                # Public Result
                '103-1679011090': 1, '103-1678831256': 0,
                '104-1678452362': 1, '104-1679222113': 0,
                '21-1673025450': 0, '21-1673032842': 0, '21-1673034242': 0, '21-1673031848': 0, '21-1673034918': 0,
                '21-1673037071': 0, '21-1673037348': 0, '21-1673037665': 0, '21-1673038169': 0, '21-1673047933': 0,
                '21-1673040188': 1, '21-1673047525': 1, '21-1673040671': 1, '21-1673041144': 1, '21-1673041994': 1,
                '21-1673043066': 1, '21-1673043367': 1, '21-1673043651': 1, '21-1673046546': 1,
                '22-1672776024': 0, '22-1672795514': 1
                }


def generate_groundtruth():
    script_path = Path(__file__).resolve().parent
    # Load Dataframe
    dataframe_res = pd.read_csv(f'{script_path}/../datasets/test.csv')

    img_paths = list(dataframe_res['img_path'])
    printer_ids = list(dataframe_res['printer_id'])
    print_id = list(dataframe_res['print_id'])
    num_files = len(img_paths)
    final_pred = []

    # Set the same label for the same exp
    for idx in range(num_files):
        exp_id = f'{printer_ids[idx]}-{print_id[idx]}'
        if ground_truth[exp_id] == 1:
            final_pred.append(1)
        else:
            final_pred.append(0)
    dataframe_res['has_under_extrusion'] = final_pred
    dataframe_res = dataframe_res.drop(['printer_id', 'print_id'], axis=1)
    dataframe_res.to_csv(f'{script_path}/../datasets/groundtruth.csv', index=False)
    return 0


def benchmark_file(filename):
    script_path = Path(__file__).resolve().parent

    dataframe_res = pd.read_csv(f'{script_path}/../datasets/{filename}.csv')
    dataframe_groundtruth = pd.read_csv(f'{script_path}/../datasets/groundtruth.csv')

    res_public = list(dataframe_res['has_under_extrusion'])[9679:]
    groundthuth_public = list(dataframe_groundtruth['has_under_extrusion'])[9679:]

    res_private = list(dataframe_res['has_under_extrusion'])[:9679]
    groundthuth_private = list(dataframe_groundtruth['has_under_extrusion'])[:9679]

    f1_public = f1_score(res_public, groundthuth_public, average='macro')
    f1_private = f1_score(res_private, groundthuth_private, average='macro')
    print(f"Public F1 Score: {f1_public}")
    print(f"Private F1 Score: {f1_private}")

    return f1_public, f1_private


if __name__ == '__main__':
    generate_groundtruth()
    benchmark_file('sample_submission')
