
from agent import PointerNet
# from environment import TSPEnvironment
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from dataset import IntegerSortDataset, sparse_seq_collate_fn

class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def masked_accuracy(output, target, mask):
    """Computes a batch accuracy with a mask (for padded sequences) """
    with torch.no_grad():
        masked_output = torch.masked_select(output, mask)
        masked_target = torch.masked_select(target, mask)
        accuracy = masked_output.eq(masked_target).float().mean()

        return accuracy

def train():
    # 環境の初期化
    # env = TSPEnvironment(num_cities=10, seed=42)

    # PointerNetの初期化
    train_set = IntegerSortDataset(num_samples=3000)
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True, num_workers=1, collate_fn=sparse_seq_collate_fn)
    test_set = IntegerSortDataset(num_samples=100)
    test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=False, num_workers=1, collate_fn=sparse_seq_collate_fn)

    model = PointerNet(input_dim=100, embedding_dim=64, hidden_dim=64)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()
    
    # モデルの訓練ループ（簡略化）
    for epoch in range(100):  # エピソード数
        model.train()  # モデルを訓練モードに設定
        for batch_idx, (seq, length, target) in enumerate(train_loader):
        
            seq, length, target = seq.to(model.device), length.cpu(), target.to(model.device)
            optimizer.zero_grad()
            log_pointer_score, argmax_pointer, mask = model(seq, length.cpu())
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), seq.size(0))
            mask = mask[:, 0, :]
            train_accuracy.update(masked_accuracy(argmax_pointer.to(model.device), target.to(model.device), mask.to(model.device)).item(), mask.int().sum().item())
            if batch_idx % 20 == 0:
                print('Epoch {}: Train [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}'
					  .format(epoch, batch_idx * len(seq), len(train_loader.dataset),
							  100. * batch_idx / len(train_loader), train_loss.avg, train_accuracy.avg))

        model.eval()
        for seq, length, target in test_loader:
            seq, length, target = seq.to(model.device), length.cpu(), target.to(model.device)

            log_pointer_score, argmax_pointer, mask = model(seq, length)
            unrolled = log_pointer_score.view(-1, log_pointer_score.size(-1))
            loss = F.nll_loss(unrolled, target.view(-1), ignore_index=-1)
            assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

            test_loss.update(loss.item(), seq.size(0))

            mask = mask[:, 0, :]
            test_accuracy.update(masked_accuracy(argmax_pointer, target, mask).item(), mask.int().sum().item())
        print('Epoch {}: Test\tLoss: {:.6f}\tAccuracy: {:.6f}'.format(epoch, test_loss.avg, test_accuracy.avg))

    print("Training completed.")

if __name__ == "__main__":
    train()

