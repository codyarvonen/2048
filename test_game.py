import unittest
import direction
import game
import numpy as np


class TestGameFunctions(unittest.TestCase):
    def setUp(self):
        self.test_game = game.Game()

        self.board = np.array([[2, 0, 0, 0], [2, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

        self.left_score = 4
        self.expected_result_move_left = np.array(
            [[2, 0, 0, 0], [4, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        self.right_score = 4
        self.expected_result_move_right = np.array(
            [[0, 0, 0, 2], [0, 0, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        self.up_score = 4
        self.expected_result_move_up = np.array(
            [[4, 2, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )
        self.down_score = 4
        self.expected_result_move_down = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 2, 2, 0]]
        )

        self.test_score_1 = 4
        self.test_row_1 = np.array([2, 2, 2, 0])
        self.expected_result_combine_tiles_1 = np.array([4, 2, 0, 0])
        self.test_score_2 = 8
        self.test_row_2 = np.array([2, 2, 2, 2])
        self.expected_result_combine_tiles_2 = np.array([4, 4, 0, 0])
        self.test_score_3 = 4
        self.test_row_3 = np.array([0, 2, 2, 2])
        self.expected_result_combine_tiles_3 = np.array([4, 2, 0, 0])
        self.test_score_4 = 4
        self.test_row_4 = np.array([2, 4, 2, 2])
        self.expected_result_combine_tiles_4 = np.array([2, 4, 4, 0])
        self.test_score_5 = 4
        self.test_row_5 = np.array([2, 0, 0, 2])
        self.expected_result_combine_tiles_5 = np.array([4, 0, 0, 0])

        self.test_board_gameover_1 = np.array(
            [[16, 8, 4, 2], [2, 4, 16, 8], [512, 2, 8, 128], [2, 4, 16, 8]]
        )
        self.test_board_gameover_2 = np.array(
            [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]]
        )
        self.test_board_not_gameover_1 = np.array(
            [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
        )
        self.test_board_not_gameover_2 = np.array(
            [[16, 8, 4, 2], [2, 4, 16, 8], [512, 512, 8, 128], [2, 4, 16, 8]]
        )
        self.test_board_not_gameover_3 = np.array(
            [[16, 8, 4, 2], [2, 4, 16, 8], [512, 0, 8, 128], [2, 4, 16, 8]]
        )

        self.empty_board = np.array(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )

        self.test_board_before_steps = np.array(
            [[4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 4]]
        )
        self.test_board_after_steps = np.array(
            [[0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, 8]]
        )
        self.test_board_step_commands = [
            direction.Direction.DOWN,
            direction.Direction.RIGHT,
        ]
        self.test_board_step_seed = 3
        self.test_step_score = 8

    def test_move_tiles(self):
        left_board, score = self.test_game.move_tiles(
            self.board.copy(), direction.Direction.LEFT
        )
        self.assertTrue(np.array_equal(self.expected_result_move_left, left_board))
        self.assertEqual(self.left_score, score)

        right_board, score = self.test_game.move_tiles(
            self.board.copy(), direction.Direction.RIGHT
        )
        self.assertTrue(np.array_equal(self.expected_result_move_right, right_board))
        self.assertEqual(self.right_score, score)

        up_board, score = self.test_game.move_tiles(
            self.board.copy(), direction.Direction.UP
        )
        self.assertTrue(np.array_equal(self.expected_result_move_up, up_board))
        self.assertEqual(self.up_score, score)

        down_board, score = self.test_game.move_tiles(
            self.board.copy(), direction.Direction.DOWN
        )
        self.assertTrue(np.array_equal(self.expected_result_move_down, down_board))
        self.assertEqual(self.down_score, score)

    def test_combine_tiles(self):
        combined_tiles, score = self.test_game.combine_tiles(self.test_row_1)
        self.assertTrue(
            np.array_equal(self.expected_result_combine_tiles_1, combined_tiles),
            f"{self.expected_result_combine_tiles_1} does not equal {combined_tiles}",
        )
        self.assertEqual(self.test_score_1, score)

        combined_tiles, score = self.test_game.combine_tiles(self.test_row_2)
        self.assertTrue(
            np.array_equal(self.expected_result_combine_tiles_2, combined_tiles),
            f"{self.expected_result_combine_tiles_2} does not equal {combined_tiles}",
        )
        self.assertEqual(self.test_score_2, score)

        combined_tiles, score = self.test_game.combine_tiles(self.test_row_3)
        self.assertTrue(
            np.array_equal(self.expected_result_combine_tiles_3, combined_tiles),
            f"{self.expected_result_combine_tiles_3} does not equal {combined_tiles}",
        )
        self.assertEqual(self.test_score_3, score)

        combined_tiles, score = self.test_game.combine_tiles(self.test_row_4)
        self.assertTrue(
            np.array_equal(self.expected_result_combine_tiles_4, combined_tiles),
            f"{self.expected_result_combine_tiles_4} does not equal {combined_tiles}",
        )
        self.assertEqual(self.test_score_4, score)

        combined_tiles, score = self.test_game.combine_tiles(self.test_row_5)
        self.assertTrue(
            np.array_equal(self.expected_result_combine_tiles_5, combined_tiles),
            f"{self.expected_result_combine_tiles_5} does not equal {combined_tiles}",
        )
        self.assertEqual(self.test_score_5, score)

    def test_add_tile(self):
        init_board = self.test_game.add_tile(self.empty_board)
        self.assertTrue(any(val == 2 or val == 4 for row in init_board for val in row))

    def test_is_game_over(self):
        game_over = self.test_game.is_game_over(self.test_board_gameover_1)
        self.assertTrue(game_over)

        game_over = self.test_game.is_game_over(self.test_board_not_gameover_1)
        self.assertFalse(game_over)

        game_over = self.test_game.is_game_over(self.test_board_gameover_2)
        self.assertTrue(game_over)

        game_over = self.test_game.is_game_over(self.test_board_not_gameover_2)
        self.assertFalse(game_over)

        game_over = self.test_game.is_game_over(self.test_board_not_gameover_3)
        self.assertFalse(game_over)

    def test_step(self):
        import random

        random_seed = random.randint(0, 2**32 - 1)
        random_game = game.Game(
            seed=random_seed, iterative_mode=True, save_game=False, visualize=False
        )
        board = random_game.init_board()

        print(random_seed)

        # Step through the game loop
        total_score = 0
        game_over = False
        count = 0
        while not game_over:
            action = random.choice(list(direction.Direction))
            print(board)
            print(action)
            board, reward, game_over = random_game.step(board, action)
            print(reward.get_total_reward(), reward.action_score)
            # total_score += reward.action_score
            # count += 1

            # board, reward, game_over = step_test_game_wrong.step(board, action)

        # step_test_game = game.Game(seed=self.test_board_step_seed)

        # result = (self.test_board_before_steps, 0, False)
        # for command in self.test_board_step_commands:
        #     result = step_test_game.step(result[0], command)

        # self.assertTrue(np.array_equal(result[0], self.test_board_after_steps))
        # self.assertEqual(result[1].action_score, self.test_step_score)
        # self.assertFalse(result[2])

        # test_board_single = np.array(
        #     [[0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [2, 0, 0, 2]]
        # )

        # test_board_single_result = np.array(
        #     [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 4, 4], [0, 0, 0, 4]]
        # )

        # step_test_game_single = game.Game(seed=4, visualize=False, iterative_mode=True)
        # result = step_test_game_single.step(
        #     test_board_single, direction.Direction.RIGHT
        # )
        # self.assertTrue(np.array_equal(result[0], test_board_single_result))
        # self.assertEqual(result[1].action_score, 4)
        # self.assertFalse(result[2])

        # new_test_seed = 2295705613
        # step_test_game_wrong = game.Game(
        #     seed=new_test_seed, iterative_mode=True, save_game=False, visualize=False
        # )
        # # step_test_game_wrong = game.Game(
        # #     seed=new_test_seed, visualize=False, iterative_mode=True
        # # )
        # board = step_test_game_wrong.init_board()
        # test_action_list = [
        #     direction.Direction.DOWN,
        #     direction.Direction.DOWN,
        #     direction.Direction.UP,
        #     direction.Direction.DOWN,
        #     direction.Direction.LEFT,
        #     direction.Direction.RIGHT,
        #     direction.Direction.LEFT,
        #     direction.Direction.DOWN,
        #     direction.Direction.DOWN,
        #     direction.Direction.DOWN,
        #     direction.Direction.UP,
        #     direction.Direction.DOWN,
        #     direction.Direction.RIGHT,
        #     direction.Direction.RIGHT,
        #     direction.Direction.RIGHT,
        #     direction.Direction.LEFT,
        # ]
        # for action in test_action_list:
        #     print(board)
        #     print(action)
        #     board, reward, game_over = step_test_game_wrong.step(board, action)
        #     print(reward.action_score)
        # print(board)
        # self.assertTrue(np.array_equal(board, test_board_single_result))
        # self.assertEqual(reward.action_score, 4)
        # self.assertFalse(game_over)


if __name__ == "__main__":
    unittest.main()
