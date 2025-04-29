# エージェントファイル
import numpy as np

gamma = 0.8


class Agent():

    ## クラス変数定義
    # アクションと移動を対応させる辞書 
    actions = ['right', 'up', 'left', 'down']
    act_dict = {'right':np.array([0,1]), 'up':np.array([-1,0]),\
                'left':np.array([0,-1]), 'down':np.array([1,0])}
    
    num_action = len(actions) # 4

    # 上下左右全て等確率でランダムに移動する
    pi_dict1 = {'right':0.25, 'up':0.25,'left':0.25, 'down':0.25} 

    def __init__(self, array_size):
        self.pos = [0, 0]   # デフォルトとして定義
        self.value_function = np.zeros(array_size)        

    # 現在位置を返す
    def get_pos(self):
        return self.pos

    # 方策
    def pi(self, state, action):
        # 変数としてstateを持っているが、実際にはstateには依存しない
        return Agent.pi_dict1[action]

    # 状態価値関数を更新する
    def update_value_function(self, action, reward):
        for i, action in enumerate(self.actions):
            reward = self.pi(self.get_pos(), action)  # Corrected to use self.get_pos() instead of q
            self.value_function = self.pi(self.get_pos(), action) * \
                                  (self.value_function + reward) * gamma
