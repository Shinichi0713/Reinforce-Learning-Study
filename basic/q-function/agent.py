# エージェントファイル
import numpy as np


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

    # 現在位置から移動
    def move(self, action):
        # 辞書を参照し、action名から移動量move_coordを取得
        move_coord = Agent.act_dict[action] 

        pos_new = self.get_pos() + move_coord
        # グリッドの外には出られない
        pos_new[0] = np.clip(pos_new[0], 0, 4)
        pos_new[1] = np.clip(pos_new[1], 0, 4)
        self.set_pos(pos_new)

    def set_pos(self, array_or_list):
        if type(array_or_list) == list:
            array = np.array(array_or_list)
        else:
            array = array_or_list
        assert (array[0] >=0 and array[0] < 5 and \
                array[1] >=0 and array[1] <5)
        self.pos = array

    # 現在位置から移動することによる報酬。この関数では移動自体は行わない
    def reward(self, state, action):
        # A地点
        if (state == np.array([0,1])).all():
            r = 10
            return r
        # B地点
        if (state == np.array([0,3])).all():
            r = 5
            return r
    
        # グリッドの境界にいて時に壁を抜けようとした時には罰則
        if (state[0] == 0 and action == 'up'):
            r = -1
        elif(state[0] == 4 and action == 'down'):
            r = -1
        elif(state[1] == 0 and action == 'left'):
            r = -1
        elif(state[1] == 4 and action == 'right'):
            r = -1
        # それ以外は報酬0
        else:
            r = 0
        return r

    def Q_pi(self, state, action,  n, out, iter_num):
        # print('\nnow entering Q_pi at n=%d, state=%s, action:%s' %(n, str(state), action))
        # state:関数呼び出し時の状態
        # n:再帰関数の呼び出し回数。関数実行時は1を指定
        # out:返り値用の変数。関数実行時は0を指定
        if n==iter_num:    # 終端状態
            out += self.pi(state, action) * self.reward(state,action)
            #print("terminal condition")
            return out
        else:
            #out += agent.reward(agent.get_pos(),action) # 報酬
            out += self.reward(state,action) # 報酬
            self.set_pos(state)
            #print("before move, pos:%s" % str(agent.get_pos()))
            self.move(action) # 移動してself.get_pos()の値が更新
            #print("after move, pos:%s" % str(agent.get_pos()))

            state_before_recursion = self.get_pos()

            ## 価値関数を再帰呼び出し
            for next_action in self.actions:
                out +=  self.pi(state_before_recursion, next_action) * \
                        self.Q_pi(state_before_recursion, next_action,  n+1, 0,  iter_num) * GAMMA
                self.set_pos(state) #  再帰先から戻ったらエージェントを元の地点に初期化
                #print("agent pos set to %s, at n:%d" % (str(state), n))
            return out