https://qiita.com/gashu/items/a9ffa292fbe1cb37bfdd

なんとこのサイト、エージェントはCNNベース


```python
MAX_SIZE = 2000
LEARNING_RATE = 0.001 
GAMMA = 0.99
EPOCH = 8
BATCH=128
EPS=0.1


transition = np.dtype([('s', np.float64, (img_stack, 96, 96)), 
                       ('a', np.float64, (3,)),
                       ('a_logp', np.float64),
                       ('r', np.float64), 
                       ('s_', np.float64, (img_stack, 96, 96))])

class Agent():
    def __init__(self, device, model):
        self.model = model.to(device)
        self.device = device
        self.buffer = np.empty(MAX_SIZE, dtype=transition)
        self.counter = 0
        self.training_step = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)  

    def select_action(self, state):
        state = state.to(device) 
        with torch.no_grad():
            alpha, beta = self.model(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == MAX_SIZE:
            self.counter = 0
            return True
        else:
            return False
        
    def update(self):
        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.float).to(self.device) #(MAX_SIZE, stack, 64, 64), torch.Size([100, 4, 64, 64])
        a = torch.tensor(self.buffer['a'], dtype=torch.float).to(self.device) #torch.Size([MAX_SIZE, 3])
        r = torch.tensor(self.buffer['r'], dtype=torch.float).to(self.device).view(-1, 1) #torch.Size([MAX_SIZE, 1])
        next_s = torch.tensor(self.buffer['s_'], dtype=torch.float).to(self.device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.float).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + GAMMA * self.model(next_s)[1] 
            adv = target_v - self.model(s)[1]

        for _ in range(EPOCH):
            for index in BatchSampler(SubsetRandomSampler(range(MAX_SIZE)), BATCH, False):
                alpha, beta = self.model(s[index])[0] 
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - EPS, 1.0 + EPS) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean() #clip付の損失の計算
                
                value_loss = F.smooth_l1_loss(self.model(s[index])[1], target_v[index]) #価値の損失計算
                entropy_loss = dist.entropy().mean() #エントロピーの計算
                loss = action_loss + 2. * value_loss - 0.01 * entropy_loss #最終の損失関数

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
```

