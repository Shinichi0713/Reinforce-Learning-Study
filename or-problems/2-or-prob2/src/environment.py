# JSSPの環境クラス
import numpy as np

class JobShopEnv:
    def __init__(self, n_jobs=5, n_machines=2):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.reset()

    # 環境リセット
    def reset(self):
        # 所要時間はランダム
        self.proc_times = np.random.randint(1, 10, size=self.n_jobs)
        # アサインされていない
        self.assigned = np.zeros(self.n_jobs, dtype=bool)
        # 各マシンの現在の空き時間
        self.machine_times = np.zeros(self.n_machines)
        self.schedule = []
        return self.proc_times.copy()

    def step(self, job, machine):
        assert not self.assigned[job]
        prev_makespan = self.machine_times.max()
        self.assigned[job] = True
        self.machine_times[machine] += self.proc_times[job]
        self.schedule.append((job, machine))
        done = self.assigned.all()
        new_makespan = self.machine_times.max()
        # 逐次報酬：makespan増加分のマイナス
        reward = -(new_makespan - prev_makespan)
        if done:
            reward += -new_makespan  # 最終的にもmakespan最小化を促す
        return self.get_state(), reward, done

    def get_state(self):
        # 各ジョブごとに [proc_time, assigned_flag] を返す
        # 各マシンの現在の空き時間も返す
        return (
            self.proc_times.copy(),             # [n_jobs]
            self.assigned.astype(np.float32),   # [n_jobs]
            self.machine_times.copy(),          # [n_machines]
        )
