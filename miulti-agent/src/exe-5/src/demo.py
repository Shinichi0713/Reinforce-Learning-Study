import torch

def run_demo_and_save_gif(env, trainer, filename="mappo_demo.gif"):
    """
    学習済みのtrainerを使用してデモを1エピソード実行し、GIFとして保存する
    """
    frames = []
    obs_list = env.reset()
    done = False
    
    # RNNの隠れ状態を初期化
    h_actors = [torch.zeros(1, 1, 128) for _ in range(env.num_agents)]
    
    # 描画用の設定
    fig, ax = plt.subplots(figsize=(6, 6))
    
    print("🎬 デモ動作を録画中...")
    
    step_count = 0
    while not done and step_count < env.max_steps:
        # 1. 現在の状態をレンダリング
        env.render(ax)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        
        # 2. 学習済みActorから行動を選択（決定論的）
        obs_tensor = torch.FloatTensor(np.array(obs_list))
        actions = []
        next_h_actors = []
        
        for i in range(env.num_agents):
            agent_id = torch.zeros(2)
            agent_id[i] = 1.0
            # 入力形式を [Batch=1, Seq=1, Dim] に整える
            a_input = torch.cat([obs_tensor[i], agent_id]).view(1, 1, -1)
            
            with torch.no_grad():
                # Actorから確率分布と隠れ状態を取得
                dist, h_a = trainer.actor(a_input, h_actors[i])
                # デモなので最も確率が高い行動を選択 (argmax)
                action = torch.argmax(dist.probs).item()
                
                actions.append(action)
                next_h_actors.append(h_a)
        
        # 3. 環境を1ステップ進める
        obs_list, rewards, done, _ = env.step(actions)
        h_actors = next_h_actors
        step_count += 1
        
    # GIFとして書き出し
    if frames:
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=250, # 1コマ 0.25秒
            loop=0
        )
        print(f"✅ デモGIFの保存が完了しました: {filename}")
    
    plt.close(fig)

# --- 実行方法 ---
# 学習ループが終わった後に呼び出してください
run_demo_and_save_gif(env, trainer, filename="mappo_final_coop_demo.gif")