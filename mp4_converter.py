from moviepy import VideoFileClip  # 修正：.editorを削除

def mp4_to_gif(input_path, output_path, fps=10):
    """
    mp4ファイルをgifに変換する
    
    Parameters:
        input_path (str): 入力mp4ファイルのパス
        output_path (str): 出力gifファイルのパス
        fps (int): 出力gifのフレームレート（デフォルト10）
    """
    # with文を使うことで、処理終了後やエラー時に自動でリソースが解放されます
    with VideoFileClip(input_path) as clip:
        
        # 必要に応じてリサイズやトリミングを追加
        # clip = clip.resized(0.5)    # 2.xでは resize -> resized に変更
        # clip = clip.subclipped(0, 5) # 2.xでは subclip -> subclipped に変更
        
        # gifとして書き出し
        clip.write_gif(output_path, fps=fps)

# 使用例
if __name__ == "__main__":
    # ファイルの存在を確認してから実行することをお勧めします
    import os
    path_input = r"D:\PycharmProjects\RL_research\maze_demo.mp4"
    dir_output = os.path.dirname(path_input)
    filename = os.path.basename(path_input).replace(".mp4", ".gif")
    path_output = os.path.join(dir_output, filename)
    if os.path.exists(path_input):
        mp4_to_gif(path_input, path_output, fps=10)
    else:
        print("エラー: input.mp4 が見つかりません。")