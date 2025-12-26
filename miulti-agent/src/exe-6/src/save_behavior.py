def save_agent_behavior_gif(agent, env, filename="agent_behavior.gif", max_steps=100):
    frames = []
    obs = env.reset()
    
    print("🎬 Generating frames for GIF...")
    
    for t in range(max_steps):
        frame = env.render_frame()
        
        # --- ここを修正：もし numpy配列なら PIL画像に変換する ---
        if isinstance(frame, np.ndarray):
            # もし [0, 1] の範囲なら 255倍するなどの処理が必要な場合があります
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            frame = Image.fromarray(frame)
        # --------------------------------------------------
        
        frames.append(frame)
        
        actions = agent.get_actions(obs, epsilon=0.0)
        next_obs, rewards, done, info = env.step(actions)
        obs = next_obs
        
        if all(done.values()):
            # 最後のフレーム処理
            last_frame = env.render_frame()
            if isinstance(last_frame, np.ndarray):
                last_frame = Image.fromarray((last_frame * 255).astype(np.uint8)) if last_frame.max() <= 1.0 else Image.fromarray(last_frame)
            frames.append(last_frame)
            print(f"✅ Goal reached in {t} steps!")
            break

    if frames:
        # frames[0] が確実に PIL Image になっているので save が使えます
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            duration=200,
            loop=0
        )
        print(f"💾 GIF saved as {filename}")