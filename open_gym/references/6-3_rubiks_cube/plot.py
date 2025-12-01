"""
overview:
    予測結果の可視化

args:
    各種パラメータ規定値は、本コード中に明記される
        - data_dir: 予測結果のリスト、list_results.pklの配置先

output:
    上記data_dirに以下の要素を出力し、標準出力に遷移サンプルを出力する
        - plot.png: 予測結果のプロット

usage-example:
    python3 plot.py --data_dir=./result
"""
import os
import argparse
import pickle
import pandas as pd

from util.visualize import set_train_plot, set_test_plot, dump_sample_seq

import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')


def get_args():
    # arg parserの設定
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./result',
                        help='directory of result pkl & history files')

    return parser.parse_args()


def main():
    # argsの取得
    args = get_args()

    # plottingの実行
    _plot(args.data_dir)
    # sample resultsを標準出力に出力
    _print(args.data_dir)


def _plot(data_dir):

    # canvasの定義
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    # training historyのプロット (plot #1)
    path = os.path.join(data_dir, 'reward_log.csv')
    with open(path, mode='r') as ofs:
        df_results = pd.read_csv(ofs)

        ax = fig.add_subplot(2, 2, 1)
        set_train_plot(ax, df_results)

    # test accuracyのプロット (plot #2)
    path = os.path.join(data_dir, 'list_results.pkl')
    with open(path, mode='rb') as ofs:
        n_obs, _, fin_flg, _, _, _, n_prd = pickle.load(ofs)

        col_name = ['n_obs', 'fin_flg', 'n_prd']
        list_summary = [_summary for _summary in zip(n_obs, fin_flg, n_prd)]
        data = pd.DataFrame(list_summary, columns=col_name)

        ax = fig.add_subplot(2, 2, 2)
        set_test_plot(ax, data)

    # plotの保存 (to data directory)
    # fig.show()
    path = os.path.join(data_dir, 'plot.png')
    fig.savefig(path)


def _print(data_dir):

    path = os.path.join(data_dir, 'list_results.pkl')
    with open(path, mode='rb') as ofs:
        [n_obs, hist_obs, fin_flg,
         st_prd, val_prd, hist_prd, n_prd] = pickle.load(ofs)

        col_name = ['n_obs', 'hist_obs', 'fin_flg', 'val_prd', 'hist_prd']
        list_summary = [_summary for _summary in zip(n_obs, hist_obs, fin_flg,
                                                     val_prd, hist_prd)]
        df = pd.DataFrame(list_summary, columns=col_name)

        list_n_scrm = range(1, 11)

        # 成功例サンプル
        print('\n### True Samples ###')
        for n_scrm in list_n_scrm:

            cond_df = df[(df.fin_flg == True) & (df.n_obs == n_scrm)]
            cond_df = cond_df.reset_index(drop=True)
            num, den = len(cond_df), len(df[df.n_obs == n_scrm])
            print('Scrambles: {0:1d}, {1:3d}/{2:3d}'.format(n_scrm, num, den))

            dump_sample_seq(cond_df, done=True, st_val=True)

        # 失敗例サンプル
        print('\n### False Samples ###')
        for n_scrm in list_n_scrm:

            cond_df = df[(df.fin_flg == False) & (df.n_obs == n_scrm)]
            cond_df = cond_df.reset_index(drop=True)
            num, den = len(cond_df), len(df[df.n_obs == n_scrm])
            print('Scrambles: {0:1d}, {1:3d}/{2:3d}'.format(n_scrm, num, den))

            dump_sample_seq(cond_df, done=False, st_val=True)


if __name__ == '__main__':
    main()
