from moviepy.editor import VideoFileClip

def mp4_to_gif(input_path, output_path, fps=10, resize=None):
    """
    MP4ファイルをGIFに変換する
    
    Parameters
    ----------
    input_path : str
        入力MP4ファイルのパス
    output_path : str
        出力GIFファイルのパス
    fps : int, optional
        出力GIFのフレームレート（デフォルト 10）
    resize : float or tuple, optional
        リサイズ比率 (例: 0.5) または (width, height)
    """
    # 動画読み込み
    clip = VideoFileClip(input_path)
    
    # 必要に応じてリサイズ
    if resize is not None:
        if isinstance(resize, (int, float)):
            clip = clip.resize(resize)
        elif isinstance(resize, tuple) and len(resize) == 2:
            clip = clip.resize(resize)
    
    # GIFとして書き出し
    clip.write_gif(output_path, fps=fps)
    
    # リソース解放
    clip.close()

# 使用例
if __name__ == "__main__":
    mp4_to_gif(
        input_path="/content/drone_test.mp4",
        output_path="drone_test.gif",
        fps=10,
        resize=0.5  # サイズを半分に縮小
    )