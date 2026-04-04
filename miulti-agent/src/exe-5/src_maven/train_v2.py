def train_episode(self):

    obs, state = self.env.reset()
    z = sample_z(1, self.z_dim)[0]

    total_loss = 0
    done = False

    while not done:

        actions = self.select_actions(obs, z, epsilon=0.3)

        reward, done, _ = self.env.step(actions)
        reward = torch.tensor([[reward]], dtype=torch.float32)

        next_obs = self.env.get_obs()
        next_state = self.env.get_state()

        qs = []
        target_qs = []

        for i in range(self.n_agents):
            o = torch.FloatTensor(obs[i]).unsqueeze(0)
            no = torch.FloatTensor(next_obs[i]).unsqueeze(0)
            z_t = z.unsqueeze(0)

            q = self.q_nets[i](o, z_t)
            next_q = self.target_q_nets[i](no, z_t)

            qs.append(q.max(1)[0])
            target_qs.append(next_q.max(1)[0])

        qs = torch.stack(qs, dim=1)
        target_qs = torch.stack(target_qs, dim=1)

        s = torch.FloatTensor(state).unsqueeze(0)
        ns = torch.FloatTensor(next_state).unsqueeze(0)

        q_tot = self.mixer(qs, s)
        target_q_tot = self.target_mixer(target_qs, ns)

        target = reward + self.gamma * target_q_tot.detach()

        loss = ((q_tot - target) ** 2).mean()

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        total_loss += loss.item()

        obs, state = next_obs, next_state

    return total_loss