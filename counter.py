import threading
import time


class LightControl():
    def __init__(self, maxTime):
        self.state = True  # 默认是开启的状态
        self.lock = threading.Lock()
        self.lastTime = time.time()  # 最后一次有人出现
        self.maxTime = maxTime  # 最长没出现时间,超出一个时间就主动关闭, 单位 s

    '''
        1. 有人过来，就开启灯光，记录最后一次的出现时间
        2. 长时间没人，就关闭
    
    '''


    def checkTime(self):
        if time.time() - self.lastTime >= self.maxTime:
            self.turnOff()
            return self.state
        else:
            return self.state

    def turnOff(self):
        with self.lock:
            self.state = False

    def turnOn(self):
        with self.lock:
            self.state = True
            self.lastTime = time.time()  # 更新最后一次出现的时间


if __name__ == '__main__':
    print(time.time())
