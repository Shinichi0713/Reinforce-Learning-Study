import os
from PIL import Image, ImageSequence

def resize_gif(input_path, output_path, scale=0.5):
    """
    既存のGIFアニメーションの解像度を変更して軽量化する関数
    :param input_path: 元のGIFファイルのパス
    :param output_path: 保存先のGIFファイルのパス
    :param scale: 縮小率（0.5 で縦横半分、面積1/4）
    """
    if not os.path.exists(input_path):
        print(f"Error: {input_path} が見つかりません。")
        return

    # GIF画像を読み込み
    with Image.open(input_path) as img:
        frames = []
        durations = []

        # 新しいサイズを計算
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)

        # 全フレームをループしてリサイズ
        for frame in ImageSequence.Iterator(img):
            # RGBAモードで変換してからリサイズ（画質劣化を防ぐため）
            resized_frame = frame.convert("RGBA").resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            frames.append(resized_frame)
            # 各フレームの表示時間を取得（デフォルト100ms）
            durations.append(frame.info.get('duration', 100))

        # 軽量化して保存
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,        # 不要なパレット情報を削除
            duration=durations,   # 元の再生速度を維持
            loop=img.info.get('loop', 0),
            disposal=2            # フレーム間の残像を防ぐ処理
        )

    # サイズの変化を表示
    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    new_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"🎬 軽量化完了:")
    print(f"  - 元ファイル: {orig_size:.2f} MB ({img.width}x{img.height})")
    print(f"  - リサイズ後: {new_size:.2f} MB ({new_width}x{new_height})")

# 実行
input_gif = r"D:\PycharmProjects\RL_research\Reinforce-Learning-Study\miulti-agent\petting_zoo\src\4_pursuit\doc\image\18_more_well_predator_v1\pursuit_mat_fixed (3).gif"
output_gif = r"D:\PycharmProjects\RL_research\Reinforce-Learning-Study\miulti-agent\petting_zoo\src\4_pursuit\doc\image\18_more_well_predator_v1\pursuit_mat_fixed_compressed.gif"

resize_gif(input_gif, output_gif, scale=0.3)