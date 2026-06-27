from PIL import Image, ImageSequence

def compress_gif_with_gifsicle(input_path, output_path, target_width=240, fps_drop_ratio=10):
    with Image.open(input_path) as im:
        frames = []
        
        # 元のGIFのループ設定を取得
        loop_setting = im.info.get('loop', 0)
        
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            # 1. フレームの間引き
            if i % fps_drop_ratio != 0:
                continue
                
            # 再生速度（duration）の取得と計算
            duration = frame.info.get('duration', 100) * fps_drop_ratio
            
            # 2. リサイズ
            if frame.width > target_width:
                aspect_ratio = frame.height / frame.width
                target_height = int(target_width * aspect_ratio)
                # 高速かつGIFに適したBILINEARまたはNEAREST（LANCZOSは情報量が増えすぎて重くなる原因になります）
                frame = frame.resize((target_width, target_height), Image.Resampling.BILINEAR)
            
            # 3. 強力な減色（色数を64色〜32色まで落とすと劇的に下がります）
            # palette=Image.Palette.ADAPTIVE で画像に最適なパレットを生成
            frame = frame.convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE, colors=64)
            
            # 新しいフレームオブジェクトにdurationを設定
            frame.info['duration'] = duration
            frames.append(frame)
            
        if not frames:
            print("エラー: フレームがありません。")
            return

        # 4. 保存（disposalをリセットし、Pillowの最適化を強制）
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,  # 共通パレット化と無駄なデータの削減
            loop=loop_setting,
            duration=[f.info.get('duration', 100) for f in frames],
            disposal=2      # 残像を防ぐ設定（環境によっては1の方が縮む場合もあります）
        )
        print(f"圧縮が完了しました: {output_path}")

# 実行例
compress_gif_with_gifsicle(r"D:\PycharmProjects\RL_research\Reinforce-Learning-Study\miulti-agent\petting_zoo\src\4_pursuit\doc\image\10_model_improvement\pursuit_mappo.gif", r"D:\PycharmProjects\RL_research\Reinforce-Learning-Study\miulti-agent\petting_zoo\src\4_pursuit\doc\image\10_model_improvement\pursuit_mappo_comp.gif")