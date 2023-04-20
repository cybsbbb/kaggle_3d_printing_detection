import argparse
import pandas as pd
import collections
import pytorch_lightning as pl
from data.data_module import ParametersDataModule
from model.network_module import ParametersClassifier
from train_config import *
from benchamark import benchmark_file

device = "cuda" if torch.cuda.is_available() else "cpu"

# Values that used for test only
script_path = Path(__file__).resolve().parent
DATA_DIR = f'{script_path}/../'
DATASET_NAME = "kaggle_contest_test"
DATA_CSV = os.path.join(DATA_DIR, "datasets/test.csv",)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-s", "--seed", default=3407, type=int, help="Set seed for training"
)
args = parser.parse_args()
seed = args.seed
set_seed(seed)


for epoch in range(11, 12):
    CHECKPOINT_PATH = f'../checkpoints/MHResAttNet-kaggle_contest_train-19042023-epoch={epoch:02d}.ckpt'
    print(f"testing checkpoint: {CHECKPOINT_PATH}")

    # Load Model from checkpoint
    model = ParametersClassifier.load_from_checkpoint(
        checkpoint_path=CHECKPOINT_PATH,
        num_classes=2,
        lr=INITIAL_LR,
        gpus=1,
        transfer=False,
    )
    model.eval()
    model.to(device)

    # Run on test dataset 20 times for voting
    model_pred_results = []
    for i in range(20):
        print(f"Voting Round: {i}")
        data = ParametersDataModule(
            batch_size=64,
            data_dir=DATA_DIR,
            csv_file=DATA_CSV,
            dataset_name=DATASET_NAME,
            has_under_extrusion=False,
        )
        data.setup('test')
        preds = []
        for batch_idx, (X, y) in enumerate(data.test_dataloader()):
            print(batch_idx)
            X = X.to(device)
            _, batch_pred = torch.max(model(X), 1)
            preds.append(batch_pred.cpu())
        model_pred = list(torch.cat(preds).numpy())
        model_pred_results.append(model_pred)

    # Get the final voting results
    model_pred_results_np = np.array(model_pred_results)
    model_pred_votes = model_pred_results_np.sum(axis=0)
    model_pred_votes = list(model_pred_votes)

    # dataframe_res = pd.read_csv(f'{script_path}/../datasets/test.csv')
    # dataframe_res['has_under_extrusion'] = model_pred_votes
    # dataframe_res = dataframe_res.drop(['printer_id', 'print_id'], axis=1)
    # dataframe_res.to_csv(f'{script_path}/../datasets/result_votes_{epoch}.csv', index=False)

    torch.cuda.empty_cache()



# dataframe_res = pd.read_csv(f'{script_path}/../datasets/test.csv')
# img_paths = list(dataframe_res['img_path'])
# printer_ids = list(dataframe_res['printer_id'])
# print_id = list(dataframe_res['print_id'])
# num_files = len(img_paths)

# counter_total = collections.defaultdict(int)
# counter_true = collections.defaultdict(int)
#
# # Stat on each exp
# for idx in range(num_files):
#     exp_id = f'{printer_ids[idx]}-{print_id[idx]}'
#     counter_total[exp_id] += 1
#     if model_pred[idx] == 1:
#         counter_true[exp_id] += 1
#
# final_pred = []
# print(len(counter_total))
# print(counter_total)
# print(counter_true)
#
# for key in counter_total:
#     print(key, counter_true[key]/counter_total[key])
#
# # Set the same label for the same exp
# for idx in range(num_files):
#     exp_id = f'{printer_ids[idx]}-{print_id[idx]}'
#     if counter_true[exp_id]/counter_total[exp_id] > 0.9:
#         final_pred.append(1)
#     else:
#         final_pred.append(0)