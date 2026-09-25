import torch
import torch.nn as nn
import torch.nn.functional as F

# --- AlphaZero風 Actor-Critic ネットワーク ---
class GoPolicyValueNet(nn.Module):
    def __init__(self):
        super(GoPolicyValueNet, self).__init__()
        
        # 共通の特徴抽出畳み込み層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Policy Head (着手確率の出力: 361次元)
        self.policy_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE)
        
        # Value Head (盤面の勝率評価: -1.0 〜 +1.0)
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x: (Batch, 3, 19, 19)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 1. Policy logits
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p) # (Batch, 361)
        
        # 2. Value
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)) # (Batch, 1)
        
        return policy_logits, value

# --- ONNXエクスポート関数 ---
def export_to_onnx(model, filename="go_agent.onnx"):
    model.eval()
    
    # ダミー入力テンソル (BatchSize=1, Channels=3, H=19, W=19)
    dummy_input = torch.randn(1, 3, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=['policy_logits', 'value'],
        dynamic_axes={
            'board_state': {0: 'batch_size'},
            'policy_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    print(f"ONNXモデルの出力が完了しました: {filename}")

if __name__ == "__main__":
    net = GoPolicyValueNet()
    export_to_onnx(net, "go_agent.onnx")