import gym

class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super(BaseWrapper, self).__init__(env)
        self.training = True

    def train(self):
        self.training = True

    def eval(self):
        self.training = False