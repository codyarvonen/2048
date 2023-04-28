import unittest
import game
import numpy as np

class TestGameFunctions(unittest.TestCase):
    def setUp(self):
        self.board = np.array([[2, 0, 0, 0], 
                               [2, 2, 2, 0], 
                               [0, 0, 0, 0], 
                               [0, 0, 0, 0]])
        self.expected_result_move_left = np.array([[2, 0, 0, 0], 
                                                   [4, 2, 0, 0], 
                                                   [0, 0, 0, 0], 
                                                   [0, 0, 0, 0]])
        self.expected_result_move_right = np.array([[0, 0, 0, 2], 
                                                    [0, 0, 2, 4], 
                                                    [0, 0, 0, 0], 
                                                    [0, 0, 0, 0]])
        self.expected_result_move_up = np.array([[4, 2, 2, 0], 
                                                 [0, 0, 0, 0], 
                                                 [0, 0, 0, 0], 
                                                 [0, 0, 0, 0]])
        self.expected_result_move_down = np.array([[0, 0, 0, 0], 
                                                   [0, 0, 0, 0], 
                                                   [0, 0, 0, 0], 
                                                   [4, 2, 2, 0]])
        
        self.test_row_1 = np.array([2, 2, 2, 0])
        self.expected_result_combine_tiles_1 = np.array([4, 2, 0, 0])
        self.test_row_2 = np.array([2, 2, 2, 2])
        self.expected_result_combine_tiles_2 = np.array([4, 4, 0, 0])
        self.test_row_3 = np.array([0, 2, 2, 2])
        self.expected_result_combine_tiles_3 = np.array([4, 2, 0, 0])
        self.test_row_4 = np.array([2, 4, 2, 2])
        self.expected_result_combine_tiles_4 = np.array([2, 4, 4, 0])
        self.test_row_5 = np.array([2, 0, 0, 2])
        self.expected_result_combine_tiles_5 = np.array([4, 0, 0, 0])

    def test_move_tiles(self):
        left_board = game.move_tiles(self.board.copy(), game.Direction.LEFT)
        self.assertTrue(np.array_equal(self.expected_result_move_left, left_board))

        right_board = game.move_tiles(self.board.copy(), game.Direction.RIGHT)
        self.assertTrue(np.array_equal(self.expected_result_move_right, right_board))

        up_board = game.move_tiles(self.board.copy(), game.Direction.UP)
        self.assertTrue(np.array_equal(self.expected_result_move_up, up_board))

        down_board = game.move_tiles(self.board.copy(), game.Direction.DOWN)
        self.assertTrue(np.array_equal(self.expected_result_move_down, down_board))


    def test_combine_tiles(self):
        combined_tiles = game.combine_tiles(self.test_row_1)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_1, combined_tiles), 
                        f'{self.expected_result_combine_tiles_1} does not equal {combined_tiles}')
        
        combined_tiles = game.combine_tiles(self.test_row_2)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_2, combined_tiles), 
                        f'{self.expected_result_combine_tiles_2} does not equal {combined_tiles}')
        
        combined_tiles = game.combine_tiles(self.test_row_3)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_3, combined_tiles), 
                        f'{self.expected_result_combine_tiles_3} does not equal {combined_tiles}')
        
        combined_tiles = game.combine_tiles(self.test_row_4)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_4, combined_tiles), 
                        f'{self.expected_result_combine_tiles_4} does not equal {combined_tiles}')
        
        combined_tiles = game.combine_tiles(self.test_row_5)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_5, combined_tiles), 
                        f'{self.expected_result_combine_tiles_5} does not equal {combined_tiles}')

    # def test_add_tile(self):
    #     self.board = [[0, 0, 0, 0],
    #                   [0, 0, 0, 0],
    #                   [0, 0, 0, 0],
    #                   [0, 0, 0, 0]]

    #     game.add_tile(self.board)
    #     self.assertTrue(any(val == 2 or val == 4 for row in self.board for val in row))

    # def test_is_game_over(self):
    #     self.board = [[2, 4, 8, 16],
    #                   [16, 8, 4, 2],
    #                   [2, 4, 8, 16],
    #                   [16, 8, 4, 2]]

    #     game_over = game.is_game_over(self.board)
    #     self.assertTrue(game_over)

    #     self.board = [[2, 4, 8, 16],
    #                   [16, 8, 4, 2],
    #                   [2, 4, 8, 16],
    #                   [0, 8, 4, 2]]

    #     game_over = game.is_game_over(self.board)
    #     self.assertFalse(game_over)

if __name__ == '__main__':
    unittest.main()
