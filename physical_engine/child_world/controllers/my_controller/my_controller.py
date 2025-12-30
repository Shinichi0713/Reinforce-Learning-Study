from controller import Robot, Motor
import math

# ロボットの初期化
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- 1. デバイス定義 ---
JOINTS = {
    'head_pan': 'PRM:/r1/c1-Joint2:11', 'head_tilt': 'PRM:/r1/c1/c2-Joint2:12',
    'tail_pan': 'PRM:/r6/c1-Joint2:61',
    'fr_hip': 'PRM:/r2/c1-Joint2:21', 'fr_knee': 'PRM:/r2/c1/c2-Joint2:22', 'fr_ankle': 'PRM:/r2/c1/c2/c3-Joint2:23',
    'fl_hip': 'PRM:/r3/c1-Joint2:31', 'fl_knee': 'PRM:/r3/c1/c2-Joint2:32', 'fl_ankle': 'PRM:/r3/c1/c2/c3-Joint2:33',
    'rr_hip': 'PRM:/r4/c1-Joint2:41', 'rr_knee': 'PRM:/r4/c1/c2-Joint2:42', 'rr_ankle': 'PRM:/r4/c1/c2/c3-Joint2:43',
    'rl_hip': 'PRM:/r5/c1-Joint2:51', 'rl_knee': 'PRM:/r5/c1/c2-Joint2:52', 'rl_ankle': 'PRM:/r5/c1/c2/c3-Joint2:53',
}

motors = {}
for name, device_id in JOINTS.items():
    device = robot.getDevice(device_id)
    if device and isinstance(device, Motor):
        device.setVelocity(20.0) # 高速走行用に制限速度を上げる(infでも可)
        device.setAvailableTorque(10.0) # トルクを十分に確保
        motors[name] = device

# --- 2. 走行パラメータ（ここを調整して速度を変えます） ---
FREQUENCY = 3.5      # 周波数を大幅にアップ (1.5 -> 3.5)
STRIDE_AMP = 0.4     # 一歩の幅を大きく (AMPLITUDE_H)
LIFT_AMP = 0.3       # 足を上げる高さを調整 (AMPLITUDE_V)
BASE_OFFSET_K = 0.6  # 基本の膝の曲がり（重心を下げる）
BASE_OFFSET_A = 0.4  # 足首のオフセット

print("Fast Run Mode Started!")

while robot.step(timestep) != -1:
    time = robot.getTime()
    
    # 進行角
    angle = 2.0 * math.pi * FREQUENCY * time
    
    # サイン波を少し加工（よりキビキビした動きにするため）
    # 基本のサイン波
    wave_a = math.sin(angle)
    wave_b = math.sin(angle + math.pi)

    # --- 脚のグループ制御 ---
    # Set A: Front-Right (FR) & Rear-Left (RL)
    # Set B: Front-Left (FL) & Rear-Right (RR)
    
    legs = [
        {'hip': 'fr_hip', 'knee': 'fr_knee', 'ankle': 'fr_ankle', 'phase': wave_a},
        {'hip': 'rl_hip', 'knee': 'rl_knee', 'ankle': 'rl_ankle', 'phase': wave_a},
        {'hip': 'fl_hip', 'knee': 'fl_knee', 'ankle': 'fl_ankle', 'phase': wave_b},
        {'hip': 'rr_hip', 'knee': 'rr_knee', 'ankle': 'rr_ankle', 'phase': wave_b},
    ]

    for leg in legs:
        m_hip = motors.get(leg['hip'])
        m_knee = motors.get(leg['knee'])
        m_ankle = motors.get(leg['ankle'])
        phase = leg['phase']

        if m_hip:
            # Hip: 前後のスイング。
            # phaseが正のとき前、負のとき後ろ。
            m_hip.setPosition(phase * STRIDE_AMP)
        
        if m_knee:
            # Knee: 走る時は、足を後ろに蹴り出す瞬間に強く曲げ、前に出す時に高く上げる。
            # phaseの正負に合わせてオフセットを調整
            knee_pos = BASE_OFFSET_K + (phase * LIFT_AMP)
            m_knee.setPosition(knee_pos)
            
        if m_ankle:
            # Ankle: 地面を蹴る動きを強調
            # 膝と連動させつつ、着地時の衝撃を逃がすように少し反転
            ankle_pos = BASE_OFFSET_A - (phase * LIFT_AMP * 0.5)
            m_ankle.setPosition(ankle_pos)

    # --- 姿勢の安定化（尻尾でバランスをとる） ---
    if 'tail_pan' in motors:
        # 脚の動きと同期させて、左右の揺れを打ち消すように振る
        motors['tail_pan'].setPosition(0.3 * math.sin(angle))