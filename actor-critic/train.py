

def learn_model(model, gamma, optimizer, device):

    R = 0
    returns = []
    for r in model.saved_rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-06)

    policy_losses = []
    value_losses = []
    for (log_prob, value), R in zip(model.saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(device)))

    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
     
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    del model.saved_rewards[:]
    del model.saved_actions[:]

    return loss
