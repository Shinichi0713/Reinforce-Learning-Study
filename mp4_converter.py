from moviepy.editor import VideoFileClip

def mp4_to_gif(input_path, output_path, fps=10):
    """
    mp4ファイルをgifに変換する
    
    Parameters:
        input_path (str): 入力mp4ファイルのパス
        output_path (str): 出力gifファイルのパス
        fps (int): 出力gifのフレームレート（デフォルト10）
    """
    # 動画を読み込み
    clip = VideoFileClip(input_path)
    
    # 必要に応じてリサイズやトリミングを追加
    # clip = clip.resize(0.5)  # サイズを半分に
    # clip = clip.subclip(0, 5)  # 0〜5秒だけ切り出す
    
    # gifとして書き出し
    clip.write_gif(output_path, fps=fps)
    
    # リソース解放
    clip.close()

# 使用例
if __name__ == "__main__":
    mp4_to_gif("input.mp4", "output.gif", fps=10)