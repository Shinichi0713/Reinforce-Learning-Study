from controller import Robot


class RobotController(Robot):
    def __init__(self):
        super().__init__()

        timestep = int(self.getBasicTimeStep())

        # ロボットに搭載されている全デバイスの数を取得
        n_devices = self.getNumberOfDevices()

        print(f"--- Device list for {self.getName()} ---")
        for i in range(n_devices):
            device = self.getDeviceByIndex(i)
            name = device.getName()
            # デバイスの型（Node Type）も取得可能
            node_type = device.getNodeType()
            print(f"Index: {i} | Name: {name} | Type ID: {node_type}")
        print("------------------------------------------")
