import os
import imageio
import gymnasium as gym
from pettingzoo.atari import wizard_of_wor_v3
from IPython.display import Image, display as ipydisplay
from pyvirtualdisplay import Display

# 1. 仮想ディスプレイの起動
if 'display' not in locals():
    display = Display(visible=0, size=(1400, 900))
    display.start()

# 2. ROMが確実に存在するパスを定義
ROM_PATH = "/usr/local/lib/python3.12/dist-packages/AutoROM/roms/"

import os
import imageio
import numpy as np
from pettingzoo.atari import wizard_of_wor_v3
from IPython.display import Image, display as ipydisplay
from pyvirtualdisplay import Display

# 仮想ディスプレイ（Colab環境用）
if 'display' not in locals():
    display = Display(visible=0, size=(1400, 900))
    display.start()

ROM_PATH = "/usr/local/lib/python3.12/dist-packages/AutoROM/roms/"

class WizardOfWorWrapper:
    def __init__(self, render_mode="rgb_array"):
        self.env = wizard_of_wor_v3.env(
            render_mode=render_mode, 
            auto_rom_install_path=ROM_PATH
        )
        self.reset()

    def reset(self):
        """環境を初期化し、最初のエージェント情報を返す"""
        self.env.reset()
        self.agent_selection = self.env.agent_selection
        return self.observe(self.agent_selection)

    def observe(self, agent):
        """特定のエージェントの現在のステータスを取得"""
        obs, reward, termination, truncation, info = self.env.last()
        return obs, reward, termination, truncation, info

    def step(self, action):
        """
        現在選択されているエージェントに行動を実行させ、
        次のエージェントの観測・報酬・フラグを返す
        """
        # 1. 行動の実行
        self.env.step(action)
        
        # 2. 次に選ばれたエージェントを更新
        self.agent_selection = self.env.agent_selection
        
        # 3. そのエージェントの最新情報を取得して返す
        return self.observe(self.agent_selection)

    def render(self):
        """現在のフレームを返す"""
        return self.env.render()

    def run_demo_gif(self, max_cycles=300, output_path="wizard_step_impl.gif"):
        """実装した step メソッドを使用してデモ走行を保存"""
        obs, reward, term, trunc, info = self.reset()
        frames = []
        
        print("シミュレーション開始（stepメソッド使用）...")
        for _ in range(max_cycles):
            # 現在のターンのエージェント
            current_agent = self.agent_selection
            
            if term or trunc:
                action = None
            else:
                # サンプル行動の取得
                action = self.env.action_space(current_agent).sample()
            
            # 実装した step を呼び出し、次のエージェントの状態を受け取る
            obs, reward, term, trunc, info = self.step(action)
            
            frames.append(self.render())
            
            if len(self.env.agents) == 0:
                break

        self.env.close()
        imageio.mimsave(output_path, frames, fps=30)
        ipydisplay(Image(filename=output_path))


# --- 実行確認 ---
wrapper = WizardOfWorWrapper()
wrapper.run_demo_gif(max_cycles=300)
