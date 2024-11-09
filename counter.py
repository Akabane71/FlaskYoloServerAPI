import threading


class LightControl():
    def __init__(self):
        self.state = True  # 默认是开启的状态
        self.lock = threading.Lock()
        self.lastTime = None  # 最后一次有人出现



