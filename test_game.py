import unittest
import game
import numpy as np

class TestGameFunctions(unittest.TestCase):
    def setUp(self):
        self.test_game = game.Game()

        self.board = np.array([[2, 0, 0, 0], 
                               [2, 2, 2, 0], 
                               [0, 0, 0, 0], 
                               [0, 0, 0, 0]])
        
        self.left_score = 4
        self.expected_result_move_left = np.array([[2, 0, 0, 0], 
                                                   [4, 2, 0, 0], 
                                                   [0, 0, 0, 0], 
                                                   [0, 0, 0, 0]])
        self.right_score = 4
        self.expected_result_move_right = np.array([[0, 0, 0, 2], 
                                                    [0, 0, 2, 4], 
                                                    [0, 0, 0, 0], 
                                                    [0, 0, 0, 0]])
        self.up_score = 4
        self.expected_result_move_up = np.array([[4, 2, 2, 0], 
                                                 [0, 0, 0, 0], 
                                                 [0, 0, 0, 0], 
                                                 [0, 0, 0, 0]])
        self.down_score = 4
        self.expected_result_move_down = np.array([[0, 0, 0, 0], 
                                                   [0, 0, 0, 0], 
                                                   [0, 0, 0, 0], 
                                                   [4, 2, 2, 0]])
        
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

        self.test_board_gameover_1 = np.array([[16, 8, 4, 2], 
                                               [2, 4, 16, 8], 
                                               [512, 2, 8, 128], 
                                               [2, 4, 16, 8]])
        self.test_board_gameover_2 = np.array([[2, 4, 2, 4], 
                                               [4, 2, 4, 2], 
                                               [2, 4, 2, 4], 
                                               [4, 2, 4, 2]])
        self.test_board_not_gameover_1 = np.array([[2, 2, 2, 2], 
                                                   [2, 2, 2, 2], 
                                                   [2, 2, 2, 2], 
                                                   [2, 2, 2, 2]])
        self.test_board_not_gameover_2 = np.array([[16, 8, 4, 2], 
                                                   [2, 4, 16, 8], 
                                                   [512, 512, 8, 128], 
                                                   [2, 4, 16, 8]])
        self.test_board_not_gameover_3 = np.array([[16, 8, 4, 2], 
                                                   [2, 4, 16, 8], 
                                                   [512, 0, 8, 128], 
                                                   [2, 4, 16, 8]])
        
        self.empty_board = np.array([[0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0],
                                     [0, 0, 0, 0]])

    def test_move_tiles(self):
        left_board, score = self.test_game.move_tiles(self.board.copy(), game.Direction.LEFT)
        self.assertTrue(np.array_equal(self.expected_result_move_left, left_board))
        self.assertEqual(self.left_score, score)

        right_board, score = self.test_game.move_tiles(self.board.copy(), game.Direction.RIGHT)
        self.assertTrue(np.array_equal(self.expected_result_move_right, right_board))
        self.assertEqual(self.right_score, score)

        up_board, score = self.test_game.move_tiles(self.board.copy(), game.Direction.UP)
        self.assertTrue(np.array_equal(self.expected_result_move_up, up_board))
        self.assertEqual(self.up_score, score)

        down_board, score = self.test_game.move_tiles(self.board.copy(), game.Direction.DOWN)
        self.assertTrue(np.array_equal(self.expected_result_move_down, down_board))
        self.assertEqual(self.down_score, score)

    def test_combine_tiles(self):
        combined_tiles, score = self.test_game.combine_tiles(self.test_row_1)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_1, combined_tiles), 
                        f'{self.expected_result_combine_tiles_1} does not equal {combined_tiles}')
        self.assertEqual(self.test_score_1, score)
        
        combined_tiles, score = self.test_game.combine_tiles(self.test_row_2)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_2, combined_tiles), 
                        f'{self.expected_result_combine_tiles_2} does not equal {combined_tiles}')
        self.assertEqual(self.test_score_2, score)
        
        combined_tiles, score = self.test_game.combine_tiles(self.test_row_3)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_3, combined_tiles), 
                        f'{self.expected_result_combine_tiles_3} does not equal {combined_tiles}')
        self.assertEqual(self.test_score_3, score)
        
        combined_tiles, score = self.test_game.combine_tiles(self.test_row_4)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_4, combined_tiles), 
                        f'{self.expected_result_combine_tiles_4} does not equal {combined_tiles}')
        self.assertEqual(self.test_score_4, score)
        
        combined_tiles, score = self.test_game.combine_tiles(self.test_row_5)
        self.assertTrue(np.array_equal(self.expected_result_combine_tiles_5, combined_tiles), 
                        f'{self.expected_result_combine_tiles_5} does not equal {combined_tiles}')
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

if __name__ == '__main__':
    unittest.main()
