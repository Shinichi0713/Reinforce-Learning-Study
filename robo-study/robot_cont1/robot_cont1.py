# from controller import Robot
from robot_controller import RobotController


# ロボットのインスタンスを作成
robot = RobotController()

# 基本のタイムステップを取得 (通常は32msや64ms)
timestep = int(robot.getBasicTimeStep())

# モーターデバイスを取得
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')

# モーターの動作モードを「速度制御モード」に設定
# 位置を infinity に設定することで、速度指定による回転が可能になります
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

# 速度を設定 (最大速度は 6.28 rad/s ですが、ここでは 2.0 に設定)
MAX_SPEED = 2.0
left_motor.setVelocity(MAX_SPEED)
right_motor.setVelocity(MAX_SPEED)

# メインループ
while robot.step(timestep) != -1:
    # ここにセンサーの読み取りなどの処理を追加できます
    print("hello")