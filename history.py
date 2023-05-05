from direction import Direction


class GameHistory():
    def __init__(self, seed: int, action_list: list[Direction]):
        self.seed = seed
        self.action_list = action_list
        self.final_board = None

    def add_action(self, action: Direction):
        self.action_list.append(action)