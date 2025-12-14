"""
overview:
    ENASを用いたネットワークアーキテクチャ検索

args:
    各種パラメータ設定値は、本コード中に明記される

output:
    RESULT_PATH に以下の要素が出力される
        - output_fig: 学習結果可視化。
        - output_txt: 出力分析に使用されるExcelファイル。
        - checkpoint: Checkpoint information file.
        - events.out.tfevents: Tensorboard file.
        - graph.pbtxt: Model protocol buffer file.
        - model.ckpt-xxx: Model checkpoint files.
        - stdout: System output logging file.

usage-example:
    python3 main.py -output_dir=./results/2019-01-01 \
    -image_dir=./data/VOC2012/JPEGImages/ \
    -label_dir=./data/VOC2012/SegmentationClass/ \
    -num_epochs=400
"""

import argparse
import os
import shutil
import sys
import time

import numpy as np
import tensorflow as tf

from agent import Agent
from environment import Environment
from utils import Log_writer, log_visualizer


# Parameters
param_dict = {}
param_dict["reset_output_dir"] = True
param_dict["log_every"] = 20
param_dict["eval_every_epochs"] = 1

param_dict["env_batch_size"] = 16
param_dict["env_num_layers"] = 12
param_dict["env_out_filters"] = 36
param_dict["env_num_operations"] = 6
param_dict["env_loss_op"] = "CE"
param_dict["env_kernel_size"] = [3, 5]
param_dict["env_dilate_rate"] = [1, 1]
param_dict["env_accuracy_op"] = "ACC"
param_dict["env_eval_batch_size"] = 100
param_dict["env_fixed_arc"] = None

param_dict["agent_train_every"] = 1
param_dict["agent_train_steps"] = 5
param_dict["agent_optim_algo"] = "adam"
param_dict["agent_lr_dec_every"] = 1000000
param_dict["agent_lstm_size"] = 64
param_dict["agent_lstm_num_layers"] = 1


def initialization(image_dir, label_dir, log_writer):
    # Initialize environment
    env = Environment(
        num_layers=param_dict["env_num_layers"],
        num_operations=param_dict["env_num_operations"],
        fixed_arc=param_dict["env_fixed_arc"],
        out_filters=param_dict["env_out_filters"],
        loss_op=param_dict["env_loss_op"],
        accuracy_op=param_dict["env_accuracy_op"],
        kernel_size=param_dict["env_kernel_size"],
        dilate_rate=param_dict["env_dilate_rate"],
        eval_batch_size=param_dict["env_eval_batch_size"],
        image_dir=image_dir,
        label_dir=label_dir,
        log_writer=log_writer
    )

    # Initialize agent
    agent = Agent(
        num_layers=param_dict["env_num_layers"],
        num_operations=param_dict["env_num_operations"],
        out_filters=param_dict["env_out_filters"],
        lstm_size=param_dict["agent_lstm_size"],
        lstm_num_layers=param_dict["agent_lstm_num_layers"],
        lr_dec_every=param_dict["agent_lr_dec_every"],
        optim_algo=param_dict["agent_optim_algo"],
        env=env,
        log_writer=log_writer)

    env.connect_agent(agent)
    agent.build_trainer()

    # Assign ops
    agent_ops = {
        "train_step": agent.train_step,
        "loss": agent.loss,
        "train_op": agent.train_op,
        "lr": agent.lr,
        "grad_norm": agent.grad_norm,
        "reward": agent.reward,
        "optimizer": agent.optimizer,
        "baseline": agent.baseline,
        "entropy": agent.sample_entropy,
        "sample_arc": agent.sample_arc,
    }

    env_ops = {
        "global_step": env.global_step,
        "loss": env.loss,
        "train_op": env.train_op,
        "lr": env.lr,
        "grad_norm": env.grad_norm,
        "train_acc": env.train_acc,
        "optimizer": env.optimizer,
        "num_train_batches": env.num_train_batches,
        "sample_arc": env.sample_arc,
        "training_init_op": env.training_init_op,
        "valid_init_op": env.valid_init_op,
        "test_init_op": env.test_init_op
    }

    ops = {
        "env": env_ops,
        "agent": agent_ops,
        "eval_every": env.num_train_batches * param_dict["eval_every_epochs"],
        "eval_func": env.eval_once,
        "num_train_batches": env.num_train_batches,
    }

    return ops


def train(output_dir, image_dir, label_dir, num_epochs, log_writer):
    # Initialization of environment and agent
    ops = initialization(image_dir, label_dir, log_writer)
    env_ops = ops["env"]
    agent_ops = ops["agent"]

    saver = tf.train.Saver(max_to_keep=2)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
        output_dir, save_steps=env_ops["num_train_batches"], saver=saver)

    hooks = [checkpoint_saver_hook]

    log_writer.print_and_write("-" * 80)
    log_writer.print_and_write("Starting session")
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(config=config_proto,
                                           hooks=hooks,
                                           checkpoint_dir=output_dir)\
            as sess:
        start_time = time.time()
        while True:
            run_ops = [
                env_ops["loss"],
                env_ops["lr"],
                env_ops["grad_norm"],
                env_ops["train_acc"],
                env_ops["train_op"]
            ]

            # Get new batch of training _build_data
            sess.run(env_ops["training_init_op"])

            # Training environment model
            loss, lr, gn, tr_acc, _ = sess.run(run_ops)
            global_step = sess.run(env_ops["global_step"])

            # Environment information logging
            actual_step = global_step
            # Increase epoch
            epoch = actual_step // ops["num_train_batches"]
            curr_time = time.time()
            if global_step % param_dict["log_every"] == 0:
                log_string = ""
                log_string += "epoch={:<6d}".format(epoch)
                log_string += "env_step={:<6d}".format(global_step)
                log_string += " loss={:<8.6f}".format(loss)
                log_string += " lr={:<8.6f}".format(lr)
                log_string += " |g|={:<8.6f}".format(gn)
                log_string += " tr_acc={:<8.6f}".format(tr_acc)
                log_string += " mins={:<10.2f}".format(
                    (curr_time - start_time) / 60)
                log_writer.print_and_write(log_string)

            # Trains agent every X step
            if actual_step % ops["eval_every"] == 0:
                if epoch % param_dict["agent_train_every"] == 0:
                    log_writer.print_and_write("Epoch {}: Training agent".format(epoch))
                    for ct_step in range(param_dict["agent_train_steps"]):
                        run_ops = [
                            agent_ops["loss"],
                            agent_ops["entropy"],
                            agent_ops["lr"],
                            agent_ops["grad_norm"],
                            agent_ops["reward"],
                            agent_ops["baseline"],
                            agent_ops["train_op"],
                        ]
                        # Training agent
                        sess.run(env_ops["valid_init_op"])
                        loss, entropy, lr, gn, val_acc, bl, _ = sess.run(
                            run_ops)
                        agent_step = sess.run(agent_ops["train_step"])

                        # Ａgent information logging
                        if ct_step % param_dict["log_every"] == 0:
                            curr_time = time.time()
                            log_string = ""
                            log_string += "agent_step={:<6d}".format(
                                agent_step)
                            log_string += " loss={:<8.6f}".format(loss)
                            log_string += " ent={:<5.2f}".format(entropy)
                            log_string += " lr={:<8.6f}".format(lr)
                            log_string += " |g|={:<8.6f}".format(gn)
                            log_string += " acc={:<8.6f}".format(val_acc)
                            log_string += " bl={:<5.2f}".format(bl)
                            log_string += " mins={:<.2f}".format(
                                float(curr_time - start_time) / 60)
                            log_writer.print_and_write(log_string)

                    log_writer.print_and_write("Here are 10 architectures")
                    arc_list = []
                    acc_list = []
                    # Sample 10 environment model structures and
                    # calculate their validation accuracy
                    for _ in range(10):
                        arc, acc = sess.run([
                            agent_ops["sample_arc"],
                            agent_ops["reward"],
                        ])
                        arc_list.append(arc)
                        acc_list.append(acc)
                        log_writer.print_and_write(arc)
                        log_writer.print_and_write("val_acc={:<8.6f}".format(acc))
                        log_writer.print_and_write("-" * 80)
                    # Display the best accuracy and structure out
                    # of the 10 sampled environment models
                    log_writer.print_and_write("best_val_acc={:<8.6f}".format(
                        np.amax(acc_list)))
                    log_writer.print_and_write("best_architecture="
                                 + str(arc_list[np.argmax(acc_list)]))
                log_writer.print_and_write("Epoch {}: Eval".format(epoch))

                # Perform validation and test on the last of
                # the 10 sampled environment models
                sess.run(env_ops["valid_init_op"])
                ops["eval_func"](sess, "valid")
                sess.run(env_ops["test_init_op"])
                ops["eval_func"](sess, "test")

            if epoch >= num_epochs:
                break


def main(args):
    try:
        output_dir = args.output_dir
        image_dir = args.image_dir
        label_dir = args.label_dir
        num_epochs = args.num_epochs
    except ValueError:
        sys.exit("Argument value invalid.")

    print("-" * 80)
    if not os.path.isdir(output_dir):
        print("Path {} does not exist. Creating.".format(output_dir))
        os.makedirs(output_dir)
    elif param_dict["reset_output_dir"]:
        print("Path {} exists. Remove and remake.".format(output_dir))
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    log_path = os.path.join(output_dir, "stdout.txt")
    log_writer = Log_writer(log_path)
    print("Logging to {}".format(log_path))

    param_writer = open(os.path.join(output_dir, "options.txt"), "w")
    for key, value in param_dict.items():
        param_writer.writelines(key + " = " + str(value) + "\n")
    param_writer.close()

    # Begin training
    train(output_dir, image_dir, label_dir, num_epochs, log_writer)

    # # Visualize log output
    log_visualizer(output_dir=output_dir,
                   num_branches=param_dict["env_num_operations"])

    log_writer.close_writer()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-output_dir",
                        help="Output files directory",
                        type=str,
                        default="results/1")
    parser.add_argument("-image_dir",
                        help="Image directory",
                        type=str,
                        default="data/VOC2012/JPEGImages/")
    parser.add_argument("-label_dir",
                        help="Label directory",
                        type=str,
                        default="data/VOC2012/SegmentationClass/")
    parser.add_argument("-num_epochs",
                        help="Number of epochs to run",
                        type=int,
                        default=400)
    args = parser.parse_args()

    main(args)
