"""
overview:
    予測結果の可視化

args:
    各種パラメータ規定値は、本コード中に明記される
        - data_dir: 予測結果のリスト、list_results.pklの配置先
        - episode_id: 可視化サンプルのepisode ID

output:
    上記data_dirに以下の要素を出力する
        - plot.png: 予測結果のプロット

usage-example:
    python3 plot.py --data_dir=./result \
    --episode_id=0
"""
import os
import argparse
import pickle
import pandas as pd

from util.visualize import set_train_plot, set_route_plot

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


def get_args():
    # arg parserの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./result',
                        help='directory of result pkl & history files')
    parser.add_argument('--episode_id', type=int, default=0,
                        help='episode id for tour samples')

    return parser.parse_args()


def main():
    # argsの取得
    args = get_args()

    # plottingの実行
    _plot(args.data_dir, args.episode_id)


def _plot(data_dir, epsd_id):

    # canvasの定義
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # training history (plot #1)のプロット
    path = os.path.join(data_dir, 'reward_log.csv')
    with open(path, mode='r') as ofs:
        df_results = pd.read_csv(ofs)

        ax = fig.add_subplot(2, 2, 1)
        set_train_plot(ax, df_results)

    # test sampleのプロット
    path = os.path.join(data_dir, 'list_results.pkl')
    with open(path, mode='rb') as ofs:
        [inputs, tours, rewards] = pickle.load(ofs)

        # 指定episodeのsampleに対する巡回路群を推定
        input, tour, reward = inputs[epsd_id], tours[epsd_id], rewards[epsd_id]

        # 推定巡回路群を重ねてプロット (plot #3)
        ax = fig.add_subplot(2, 2, 3)
        set_route_plot(ax, input, tour, reward, best_flg=False)

        # 最短推定巡回路のプロット (plot #4)
        ax = fig.add_subplot(2, 2, 4)
        set_route_plot(ax, input, tour, reward, best_flg=True)

    # plotの保存 (to data directory)
    # fig.show()
    path = os.path.join(data_dir, 'plot.png')
    fig.savefig(path)


if __name__ == '__main__':
    main()
