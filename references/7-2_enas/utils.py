# coding: utf-8

from collections import defaultdict
import argparse
import decimal
import os
import sys

import matplotlib
matplotlib.use("Agg")   # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pylab


decimal.getcontext().prec = 6


class Log_writer:
    def __init__(self,
                 log_path):

        self.log_path = log_path
        self.writer = open(log_path, "w")

    def print_and_write(self, message):
        print(message)
        self.writer.write(str(message) + "\n")
        self.writer.flush()

    def close_writer(self):
        self.writer.close()


def log_visualizer(
        output_dir="./",
        output_csv="output_txt",
        output_fig="output_fig",
        input_log="stdout.txt",
        num_branches=6,
        env_loss="loss",
        env_step="env_step",
        env_reward="tr_acc",
        agent_loss="loss",
        agent_step="agent_step",
        agent_reward="acc",):

    output_csv = os.path.join(output_dir, output_csv)
    output_fig = os.path.join(output_dir, output_fig)
    os.makedirs(output_csv, exist_ok=True)
    os.makedirs(output_fig, exist_ok=True)

    log_dict = {}
    log_dict = defaultdict(lambda: [], log_dict)

    # Convert the log file into a dictionary.
    f = open(os.path.join(output_dir, input_log))
    count = 0
    line = f.readline()
    while line:
        line = line.rstrip("\n")
        if line == "Starting session":
            count += 1
        else:
            # Split a log line into keys and values.
            line = line.replace(": ", "=").replace("=", " ")
            line = line.replace("[", "").replace("]", "")
            temp = line.split()
            # Append the log line contents into a dictionary.
            if len(temp) == 2:
                log_dict[temp[0]].append(temp[1])
            elif len(temp) > 2:
                if temp[0] == "best_architecture":
                    log_dict[temp[0]].append(",".join(temp[1:len(temp)]))
                elif len(temp) % 2 == 0:
                    temp_dict = {}
                    for i in range(len(temp) // 2):
                        temp_dict[temp[2 * i]] = temp[2 * i + 1]
                    log_dict[temp[0]].append(temp_dict)
        line = f.readline()
    f.close()

    # Check data integrity
    checklist = [log_dict["best_architecture"],
                   log_dict["best_val_acc"],
                   log_dict["valid_accuracy"],
                   log_dict["test_accuracy"]]
    num_top = min([len(log_dict["best_architecture"]),
                   len(log_dict["best_val_acc"]),
                   len(log_dict["valid_accuracy"]),
                   len(log_dict["test_accuracy"])])
    if num_top <= 0:
        sys.exit("Not enough data to execute visualization.")
    for item in checklist:
        while len(item) > num_top:
            item.pop(-1)
    if num_top > 10:
        num_top = 10

    # The log output of env model training.
    env_df = pd.io.json.json_normalize(log_dict,
                                       record_path="epoch").astype(np.float64)

    # The log output of agent model training.
    agent_df = pd.io.json.json_normalize(log_dict,
                                         record_path="agent_step"
                                         ).astype(np.float64)

    # The log output of train/val/test reward.
    acc_df = pd.DataFrame({"best_val_acc": log_dict["best_val_acc"],
                           "val_acc": log_dict["valid_accuracy"],
                           "test_acc": log_dict["test_accuracy"]}
                          ).astype(np.float64)
    acc_df = acc_df.loc[:, ["best_val_acc", "val_acc", "test_acc"]]

    # The log output of the best architecture in each epoch.
    arc_df = pd.DataFrame({"best_val_acc": log_dict["best_val_acc"],
                           "best_architecture": log_dict["best_architecture"]}
                          )

    arc_df = arc_df.loc[:, ["best_val_acc", "best_architecture"]]
    arc_df["best_val_acc"] = arc_df["best_val_acc"].astype(np.float64)
    arc_df = arc_df.sort_values("best_val_acc", ascending=False).reset_index()

    keys = [str(t) for t in range(num_branches)]
    values = [0 for t in range(num_branches)]
    temp = dict(zip(keys, values))
    temp_height = []

    temp_list = arc_df["best_architecture"]
    for i in range(len(temp_list)):
        temp_seq = temp_list[i].split(",")
        for j in temp_seq:
            temp[str(j)] += 1
        temp_s = sorted(temp.items(), key=lambda x: x[0])
        height = []
        for k in range(len(temp_s)):
            height.append(temp_s[k][1])
        height = [decimal.Decimal(float(s))
                  / decimal.Decimal(12 * (i + 1)) for s in height]
        temp_height.append(height)

    height_df = pd.DataFrame(temp_height)
    arc_df = pd.concat([arc_df, height_df], axis=1)

    # Save each log output as a csv file.
    env_df.to_csv(os.path.join(output_csv, "env.csv"))
    agent_df.to_csv(os.path.join(output_csv, "agent.csv"))
    acc_df.to_csv(os.path.join(output_csv, "reward_development.csv"))
    arc_df.to_csv(os.path.join(output_csv, "best_architecture.csv"))

    # Barplot branch distribution of the high score architectures.
    left = [str(t + 1) for t in range(num_top)[::-1]]
    pylab.rcParams["font.size"] = 12
    plt.figure(figsize=(1.33 * num_branches, 4 * num_top / 10))
    plt.subplots_adjust(wspace=0.1)

    for i in range(num_branches):
        plt.subplot(1, 6, i + 1)
        temp_list = list(arc_df.iloc[0:num_top, 3 + i])[::-1]
        plt.barh(left, temp_list, align="edge", height=0.7, color="#d62728")
        plt.grid(True)
        plt.xlim([0, 0.35])
        plt.title("branch" + str(i))
        if i > 0:
            plt.tick_params(labelbottom=False, bottom=False)
            plt.tick_params(labelleft=False, left=False)

    plt.savefig(os.path.join(output_fig, "branch_distribution.png"))

    # Plot log output of train/val/test reward.
    pylab.rcParams["font.size"] = 12
    plt.figure(figsize=(6, 4))

    plots1 = plt.plot(acc_df)
    plt.legend(plots1, ("best_valid", "valid", "test"),
               loc="best",  # Legend location.
               framealpha=0.25,  # Legend background alpha transparency.
               prop={"size": "small",
                     "family": "monospace"})  # Legend font properties.
    plt.title("Reward Development Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_fig, "reward_development.png"))

    # Plot log output of env model training.
    pylab.rcParams["font.size"] = 12
    plt.figure(figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.plot(env_df[env_step], env_df[env_reward], color="#ff7f0e")
    plt.title("Reward Development Graph: Env Model")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(env_df[env_step], env_df[env_loss])
    plt.title("Loss Development Graph: Env Model")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_fig, "env_development.png"))

    # Plot log output of agent model training.
    pylab.rcParams["font.size"] = 12
    plt.figure(figsize=(6, 8))

    plt.subplot(2, 1, 1)
    plt.plot(agent_df[agent_step], agent_df[agent_reward], color="#ff7f0e")
    plt.title("Reward Development Graph: Agent Model")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(agent_df[agent_step], agent_df[agent_loss])
    plt.title("Loss Development Graph: Agent Model")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(output_fig, "agent_development.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir",
                        help="Output files directory",
                        type=str,
                        default="results/1")
    args = parser.parse_args()
    output_dir = args.output_dir

    log_visualizer(output_dir)
