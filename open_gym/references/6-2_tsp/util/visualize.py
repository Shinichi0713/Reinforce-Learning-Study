import numpy as np


def set_train_plot(ax, df):
    if 'episode' not in df.keys():
        return
    if 'avg_reward' not in df.keys():
        return
    if 'avg_loss' not in df.keys():
        return

    loss_scale_1 = 10
    # loss_scale_2 = 1
    loss_scale_3 = 1
    loss_scale_4 = 10

    x = df['episode']
    y1 = df['avg_reward'] / loss_scale_1
    # y2 = df['avg_loss'] / loss_scale_2
    y3 = df['avg_vloss'] / loss_scale_3
    y4 = df['avg_aloss'] / loss_scale_4

    kwargs = {'alpha': 0.5}
    ax.plot(x, y1, 'r-',
            label='avg_reward/episode (x1/{})'.format(loss_scale_1), **kwargs)
    # ax.plot(x, y2, 'b:',
    #         label='avg_loss/episode (x1/{})'.format(loss_scale_2), **kwargs)
    ax.plot(x, y3, 'g-',
            label='avg_vloss/episode (x1/{})'.format(loss_scale_3), **kwargs)
    ax.plot(x, y4, 'b-',
            label='avg_ploss/episode (x1/{})'.format(loss_scale_4), **kwargs)

    ax.set_ylim(-0.6, 0.5)

    ax.legend(loc='best')

    ax.set_title('training history')
    ax.set_xlabel('number of episodes')
    ax.set_ylabel('amplitude')


def set_route_plot(ax, batch_input, batch_tour, batch_reward, best_flg):

    # plot cities
    ax.scatter(batch_input[0, :, 0], batch_input[0, :, 1], c='r', s=30)

    # plot prd tours
    if not best_flg:

        for i_batch, (input, tour) in enumerate(zip(batch_input, batch_tour)):

            if i_batch % 1:  # 1, 5, 10
                continue

            # plot tours
            X = input[tour, 0]
            Y = input[tour, 1]

            _label = '' if i_batch else 'tours'
            ax.plot(X, Y, 'k-', alpha=0.1, label=_label)

    # plot best route
    if best_flg:

        # best tour
        best_batch = np.argmax(batch_reward)
        best_length = -1.0 * 100. * np.max(batch_reward)

        input = batch_input[best_batch]
        tour = batch_tour[best_batch]

        # plot best tour
        X = input[tour, 0]
        Y = input[tour, 1]

        ax.plot(X, Y, 'r-', alpha=0.8, label='best tour')
        print('best tour length: {:3.3f}'.format(best_length))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.legend(loc='best')

    ax.set_title('predicted route')
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('y coordinate')
