# !pip install -q onnx onnxscript

# ---------------------------------------------------------
# 3. 学習と ONNX エクスポート
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CnnTransformerHybrid().to(device)

# CIFAR-10 データセット設定
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# 簡易学習（2エポック程度で精度向上を確認）
model.train()
for epoch in range(2):
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

# ONNX 形式へのエクスポート
model.eval().cpu()
dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32)

torch.onnx.export(
    model,
    dummy_input,
    "cnn_cifar10.onnx", # 前の C++ コードとファイル名を一致
    export_params=True,
    opset_version=14,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("\nONNXモデルを出力しました: cnn_cifar10.onnx")


import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

# 1. CIFAR-10 のテストデータセットをロード（Transformなしで生の画像データ（PIL Image）を取得）
testset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True
)

# CIFAR-10 のクラス名一覧
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

# 2. テストデータから 1 枚選択（例: インデックス 0 番目の画像）
image_idx = 0
pil_img, label = testset[image_idx]

print(f"Selected image index: {image_idx}")
print(f"Ground Truth Label: {label} ({classes[label]})")

# 3. OpenCV 形式 (BGR) に変換して保存
# PIL (RGB) -> NumPy array (RGB) -> BGR
img_np = np.array(pil_img)
img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

# 画像を保存
cv2.imwrite("test_image.png", img_bgr)
print("Saved real test image as 'test_image.png'")