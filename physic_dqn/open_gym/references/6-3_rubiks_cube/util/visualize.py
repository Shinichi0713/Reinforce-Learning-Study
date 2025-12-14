import numpy as np
import pandas as pd


def set_train_plot(ax, df):
    # TODO: too redundant, simplify
    if 'episode' not in df.keys():
        return
    if 'avg_reward' not in df.keys():
        return
    if 'avg_loss' not in df.keys():
        return

    loss_scale = 10

    x = df['episode']
    y1 = df['avg_reward']
    y2 = df['avg_loss'] / loss_scale

    kwargs = {'alpha': 0.5}
    ax.plot(x, y1, 'r-',
            label='avg_reward/episode', **kwargs)
    ax.plot(x, y2, 'b:',
            label='avg_loss/episode (x1/{})'.format(loss_scale), **kwargs)

    ax.legend(loc='best')

    ax.set_title('training history')
    ax.set_xlabel('number of episodes')
    ax.set_ylabel('amplitude')


def set_test_plot(ax, df, label='', col='k'):

    # obtain entries
    # fin_flg = df.fin_flg
    fin_flg = df.fin_flg & (df.n_obs >= df.n_prd)

    df1 = pd.DataFrame([1 for itr in df.n_obs], index=df.n_obs)
    df2 = pd.DataFrame([1 if itr else 0 for itr in fin_flg], index=df.n_obs)
    sr1, sr2 = df1.groupby('n_obs').sum(), df2.groupby('n_obs').sum()
    bin, frq1, frq2 = list(sr1.index), sr1.T.values[0], sr2.T.values[0]

    # cent value
    x = bin
    y = [frq2[i] / frq1[i] for i in range(len(frq1))]

    # apply polyfit
    n = 4  # 2, 3, 4, 5
    par = np.polyfit(x, y, n)
    ply = np.poly1d(par)

    # drawing
    _fmt_str = col + ':'
    ax.plot(x, [1.0]*len(x), _fmt_str, alpha=0.5)
    _fmt_str = col + '--'
    ax.plot(x, ply(x), _fmt_str, label=label)
    _fmt_str = col + 'o'
    ax.plot(x, y, _fmt_str)

    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 1.01)

    if label:
            ax.legend(loc='best')

    ax.set_title('completion ratio dependence')
    ax.set_xlabel('number of scrambles')
    ax.set_ylabel('completion ratio')


def dump_sample_seq(df, done=True, st_val=False):

        if not len(df):
            print(' No Samples')
            return

        data_id = np.random.choice(list(df.index), 1)
        data_id = data_id[0]

        scrm_hist = df['hist_obs'][data_id]

        log_str = ' G | '
        for scrm in scrm_hist:
            log_str += '--['
            log_str += str(scrm)
            log_str += ']--> '
        log_str += '| S'
        print(log_str)

        act_hist = df['hist_prd'][data_id]
        val_hist = df['val_prd'][data_id]
        inv_act_hist = act_hist[::-1]
        inv_val_hist = val_hist[::-1]

        log_str = ' G |' if done else ' *'
        for act, val in zip(inv_act_hist, inv_val_hist[1:]):
            log_str += ' <--['
            log_str += str(act)
            log_str += ']--'
            if st_val:
                log_str += ' ({:2.2f})'.format(val[0])
        log_str += ' | S'
        print(log_str)
