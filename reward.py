from typing import Tuple
import numpy as np

# from game import Game
from direction import Direction


class ActionReward:
    def __init__(self, action_score: int, old_board: np.ndarray, new_board: np.ndarray):
        self.action_score = action_score
        self.old_board = old_board
        self.new_board = new_board

    def get_largest_tile_score(self):
        if np.amax(self.new_board) > np.amax(self.old_board):
            return np.amax(self.new_board)
            # return 2
        else:
            return 1

    def get_capacity_factor(self):
        capacity_factor = np.count_nonzero(self.new_board) / np.size(self.new_board)
        return 1 - capacity_factor

    def get_proximity_factor(self):
        # Calculate the distances between adjacent tiles
        distances = 0
        for i in range(3):
            for j in range(3):
                # Calculate the distance between adjacent tiles horizontally
                distances += abs(self.new_board[i][j] - self.new_board[i][j + 1])
                # Calculate the distance between adjacent tiles vertically
                distances += abs(self.new_board[i][j] - self.new_board[i + 1][j])
        for i in range(3):
            # Calculate the distance between the right-most column and the adjacent column
            distances += abs(self.new_board[i][3] - self.new_board[i + 1][3])
        for j in range(3):
            # Calculate the distance between the bottom row and the adjacent row
            distances += abs(self.new_board[3][j] - self.new_board[3][j + 1])

        # Calculate the proximity score
        max_val = np.max(self.new_board)
        max_distance = max_val * 2 * 16
        # min_val = np.min(self.new_board)

        if (
            max_val == self.new_board[0][0]
            or max_val == self.new_board[0][3]
            or max_val == self.new_board[3][0]
            or max_val == self.new_board[3][3]
        ):
            # High numbered tiles are in corners, so proximity score should be high
            proximity = distances / 4.0
        elif (
            max_val == self.new_board[0][1]
            or max_val == self.new_board[0][2]
            or max_val == self.new_board[1][0]
            or max_val == self.new_board[2][0]
            or max_val == self.new_board[1][3]
            or max_val == self.new_board[2][3]
            or max_val == self.new_board[3][1]
            or max_val == self.new_board[3][2]
        ):
            # High numbered tiles are on edges, so proximity score should be medium
            proximity = distances / 2.0
        else:
            # High numbered tiles are in the middle, so proximity score should be low
            proximity = distances

        proximity = 1 - (proximity / max_distance)
        return proximity

    # TODO: refactor these operations somehow
    def combine_tiles(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        score = 0
        arr = arr[arr != 0]
        for i in range(len(arr) - 1):
            if arr[i] == arr[i + 1]:
                arr[i] *= 2
                score += arr[i]
                arr[i + 1] = 0
        arr = arr[arr != 0]
        arr = np.concatenate(
            (arr, np.zeros(self.old_board.shape[0] - len(arr), dtype=int))
        )
        return arr, score

    def move_tiles(
        self, board: np.ndarray, direction: Direction
    ) -> Tuple[np.ndarray, int]:
        score = 0
        rotated_board = board.copy()
        rotated_board = np.rot90(rotated_board, direction.value)
        for i in range(rotated_board.shape[0]):
            arr = rotated_board[i, :]
            arr = np.trim_zeros(arr)
            arr, row_score = self.combine_tiles(arr)
            score += row_score
            rotated_board[i, :] = arr
        new_board = np.rot90(rotated_board, -direction.value)
        return new_board, score

    def get_availability_factor(self):
        # test_game = Game()
        results = []
        results.append(self.move_tiles(self.new_board, Direction.LEFT))
        results.append(self.move_tiles(self.new_board, Direction.RIGHT))
        results.append(self.move_tiles(self.new_board, Direction.UP))
        results.append(self.move_tiles(self.new_board, Direction.DOWN))

        available_directions = 0
        for result in results:
            if not np.array_equal(result[0], self.new_board):
                available_directions += 1

        return available_directions / 4

    def get_invalid_move_factor(self):
        if np.array_equal(self.new_board, self.old_board):
            return -10
        else:
            return 1

    def future_merge_available(self, board) -> bool:
        # Check rows
        if (board[:, :-1] == board[:, 1:]).any():
            return True
        # Check columns
        if (board[:-1, :] == board[1:, :]).any():
            return True
        # No adjacent numbers found
        return False

    def get_total_reward(self):
        invalid_factor = self.get_invalid_move_factor()
        if invalid_factor == -10:
            return invalid_factor
        if self.action_score == 0 and self.future_merge_available(self.new_board):
            self.action_score = 1
        return (
            self.action_score
            * self.get_availability_factor()
            * self.get_capacity_factor()
            * invalid_factor
            * self.get_largest_tile_score()
            # self.get_proximity_factor() *
        )

    # TODO: Add reward test functions!!!!
