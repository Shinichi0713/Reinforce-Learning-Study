from controller import Robot, Motor
import math

# ロボットの初期化
robot = Robot()

# 基本タイムステップの取得
timestep = int(robot.getBasicTimeStep())
print("Aibo Trot Controller started!")

# --- 1. デバイス名定義 (ユーザー提供リストより) ---
# ※ この定義ブロックは、コードの先頭（importとrobot()の間）に移動し、
#    一度定義されたJOINTS辞書を再利用できるようにするのが理想です。
JOINTS = {
    # 頭と尻尾
    'head_pan': 'PRM:/r1/c1-Joint2:11',
    'head_tilt': 'PRM:/r1/c1/c2-Joint2:12',
    'tail_pan': 'PRM:/r6/c1-Joint2:61',
    'camera': 'PRM:/r1/c1/c2/c3/i1-FbkImageSensor:F1',

    # 前右 (FR) - r2
    'fr_hip': 'PRM:/r2/c1-Joint2:21',
    'fr_knee': 'PRM:/r2/c1/c2-Joint2:22',
    'fr_ankle': 'PRM:/r2/c1/c2/c3-Joint2:23',
    
    # 前左 (FL) - r3
    'fl_hip': 'PRM:/r3/c1-Joint2:31',
    'fl_knee': 'PRM:/r3/c1/c2-Joint2:32',
    'fl_ankle': 'PRM:/r3/c1/c2/c3-Joint2:33',
    
    # 後右 (RR) - r4
    'rr_hip': 'PRM:/r4/c1-Joint2:41',
    'rr_knee': 'PRM:/r4/c1/c2-Joint2:42',
    'rr_ankle': 'PRM:/r4/c1/c2/c3-Joint2:43',
    
    # 後左 (RL) - r5
    'rl_hip': 'PRM:/r5/c1-Joint2:51',
    'rl_knee': 'PRM:/r5/c1/c2-Joint2:52',
    'rl_ankle': 'PRM:/r5/c1/c2/c3-Joint2:53',
}

# --- 2. デバイスの取得と初期設定 ---
# このブロックはループ外で一度だけ実行されるべきです。
# ロボットの初期化コードと統合しました。
motors = {}
camera = None
for name, device_id in JOINTS.items():
    device = robot.getDevice(device_id)
    if device:
        if 'camera' in name:
            camera = device # カメラはMotorではない
        else:
            motor = device
            # モーターの初期設定
            motor.setVelocity(float('inf')) 
            motor.setForce(100.0) 
            motors[name] = motor
    else:
        print(f"ERROR: Device {device_id} not found.")


# --- 2. カメラの有効化 ---
if camera is not None:
    camera.enable(timestep)
    print(f"カメラを有効化しました: {JOINTS['camera']}")
else:
    print("WARNING: カメラが見つかりませんでした。")

# --- 3. 歩行パラメータの設定 (定数としてループ外に保持) ---
FREQUENCY = 1.5   # 歩行頻度 (Hz)
AMPLITUDE_H = 0.1 # 水平方向（Hip）の振幅 (rad)
AMPLITUDE_V = 0.2 # 垂直方向（Knee/Ankle）の振幅 (rad)
OFFSET_V = 0.5    # 垂直方向のオフセット（足を上げるための初期角度）
HEAD_SWING = 0.05 # 首の振幅 (rad)
TAIL_SWING_AMP = 0.15
TAIL_SWING_OFF = 0.15

# --- 4. メインループ (歩行ロジック) ---
print("Aibo controller started! 簡易トロット歩行を実行します。")
while robot.step(timestep) != -1:
    time = robot.getTime()
    
    # 進行角 (サイン波の引数)
    angle = 2.0 * math.pi * FREQUENCY * time
    
    # トロット歩行の実現 (対角線セット)
    set_a_phase = math.sin(angle)
    set_b_phase = math.sin(angle + math.pi) # 180度 (π) ずらす

    
    # --- A. 脚の動作制御 (FR & RL) ---
    # セットAの関節名リスト
    set_a_hips = [motors.get('fr_hip'), motors.get('rl_hip')]
    set_a_knees = [motors.get('fr_knee'), motors.get('rl_knee')]
    set_a_ankles = [motors.get('fr_ankle'), motors.get('rl_ankle')] # 追加

    for hip, knee, ankle in zip(set_a_hips, set_a_knees, set_a_ankles):
        if hip:
            # Hip: 前後スイング
            hip.setPosition(set_a_phase * AMPLITUDE_H)
        
        if knee:
            # Knee & Ankle: 足を持ち上げて前に出す
            knee.setPosition(set_a_phase * AMPLITUDE_V + OFFSET_V)
            
        if ankle:
            # Ankle: 安定化のためKneeと逆方向に動かすなど、調整が必要
            # ここではKneeの動きを反転させて設定
            ankle.setPosition(-set_a_phase * AMPLITUDE_V + OFFSET_V) 


    # --- B. 脚の動作制御 (FL & RR) ---
    # セットBの関節名リスト
    set_b_hips = [motors.get('fl_hip'), motors.get('rr_hip')]
    set_b_knees = [motors.get('fl_knee'), motors.get('rr_knee')]
    set_b_ankles = [motors.get('fl_ankle'), motors.get('rr_ankle')] # 追加

    for hip, knee, ankle in zip(set_b_hips, set_b_knees, set_b_ankles):
        if hip:
            # Hip: 前後スイング
            hip.setPosition(set_b_phase * AMPLITUDE_H)

        if knee:
            # Knee & Ankle: 足を持ち上げて前に出す
            knee.setPosition(set_b_phase * AMPLITUDE_V + OFFSET_V)

        if ankle:
            # Ankle: 安定化のためKneeの動きを反転
            ankle.setPosition(-set_b_phase * AMPLITUDE_V + OFFSET_V) 


    # --- C. 頭と尻尾の制御 (一本化) ---
    if 'head_pan' in motors:
        motors['head_pan'].setPosition(HEAD_SWING * math.sin(2.0 * time))

    if 'tail_pan' in motors:
        tail_pos = TAIL_SWING_AMP * math.sin(10.0 * time) + TAIL_SWING_OFF
        motors['tail_pan'].setPosition(tail_pos)