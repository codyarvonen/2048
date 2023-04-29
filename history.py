from direction import Direction


class GameHistory():
    def __init__(self, seed: int, action_list: list[Direction]):
        self.seed = seed
        self.action_list = action_list