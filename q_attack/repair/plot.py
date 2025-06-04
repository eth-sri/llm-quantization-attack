import json
import os

import numpy as np

from q_attack.repair.convert import convert1, convert2
from q_attack.repair.parser import parse_args_plot
from q_attack.helpers.plot_func import my_plot
from q_attack.helpers.train_torch import LogEachType, LogHistory

WINDOW_SIZE = 500  # used for per-iteration log


def build_acc_list(log_each: list[LogEachType], window_size=1):
    x, y = [], []
    for i in range(0, len(log_each) - window_size + 1, window_size):
        x.append(np.mean([log.epoch for log in log_each[i : i + window_size]]))
        y.append(np.mean([log.correct_backdoor / max(1, log.total) for log in log_each[i : i + window_size]]))
    return x, y


def build_loss_list(log_each: list[LogEachType], window_size=1):
    x, y = [], []
    for i in range(0, len(log_each) - window_size + 1, window_size):
        x.append(np.mean([log.epoch for log in log_each[i : i + window_size]]))
        y.append(np.mean([log.loss_sum_backdoor / max(1, log.total) for log in log_each[i : i + window_size]]))

    return x, y


def build_accuracy_plot_list(log_history: LogHistory):
    return_list = []
    acc_train_clean_x, acc_train_clean_y = build_acc_list(log_history.train_clean, window_size=WINDOW_SIZE)
    return_list.append(
        [acc_train_clean_x, acc_train_clean_y, {"label": "train clean acc", "linewidth": 0.5, "alpha": 0.6}]
    )

    acc_train_backdoor_x, acc_train_backdoor_y = build_acc_list(
        log_history.train_backdoor_label_flipped, window_size=WINDOW_SIZE
    )
    return_list.append(
        [acc_train_backdoor_x, acc_train_backdoor_y, {"label": "train backdoor acc", "linewidth": 0.5, "alpha": 0.6}]
    )

    acc_val_clean_x, acc_val_clean_y = build_acc_list(log_history.val_clean)
    return_list.append(
        [
            acc_val_clean_x,
            acc_val_clean_y,
            {"label": "val clean acc", "marker": ".", "markersize": 10, "alpha": 0.6},
        ]
    )

    acc_val_backdoor_x, acc_val_backdoor_y = build_acc_list(log_history.val_backdoor_label_flipped)
    return_list.append(
        [
            acc_val_backdoor_x,
            acc_val_backdoor_y,
            {"label": "val backdoor acc", "marker": ".", "markersize": 10, "alpha": 0.6},
        ]
    )

    return return_list


def build_loss_plot_list(log_history: LogHistory):
    return_list = []
    loss_train_clean_x, loss_train_clean_y = build_loss_list(log_history.train_clean, window_size=WINDOW_SIZE)
    return_list.append(
        [loss_train_clean_x, loss_train_clean_y, {"label": "train clean loss", "linewidth": 0.5, "alpha": 0.6}]
    )

    loss_train_backdoor_x, loss_train_backdoor_y = build_loss_list(
        log_history.train_backdoor_label_flipped, window_size=WINDOW_SIZE
    )
    return_list.append(
        [loss_train_backdoor_x, loss_train_backdoor_y, {"label": "train backdoor loss", "linewidth": 0.5, "alpha": 0.6}]
    )

    loss_val_clean_x, loss_val_clean_y = build_loss_list(log_history.val_clean)
    return_list.append(
        [
            loss_val_clean_x,
            loss_val_clean_y,
            {"label": "val clean loss", "marker": ".", "markersize": 10, "alpha": 0.6},
        ]
    )

    loss_val_backdoor_x, loss_val_backdoor_y = build_loss_list(log_history.val_backdoor_label_flipped)
    return_list.append(
        [
            loss_val_backdoor_x,
            loss_val_backdoor_y,
            {"label": "val backdoor loss", "marker": ".", "markersize": 10, "alpha": 0.6},
        ]
    )

    return return_list


def load_log_history(json_path: str, version=0):
    with open(json_path, "r") as f:
        json_log_history = json.load(f)
    if version == 1:
        json_log_history = convert1(json_log_history)
    elif version == 2:
        json_log_history = convert2(json_log_history)
        print(json_log_history["train_clean"][0])
    for key in json_log_history.keys():
        json_log_history[key] = [LogEachType(**log) for log in json_log_history[key]]

    log_history = LogHistory(**json_log_history)
    return log_history


def main():
    args = parse_args_plot()
    log_history = load_log_history(os.path.join(args.output_dir, "log_history.json"), version=2)

    acc_plot_list = build_accuracy_plot_list(log_history)
    my_plot(acc_plot_list, xlabel="epoch", ylabel="acc", title="acc", output_path=f"{args.output_dir}/accuracy.png")
    loss_plot_list = build_loss_plot_list(log_history)
    my_plot(loss_plot_list, xlabel="epoch", ylabel="loss", title="loss", output_path=f"{args.output_dir}/loss.png")


if __name__ == "__main__":
    main()
