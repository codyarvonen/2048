import time
from typing import Tuple
import pygame
import numpy as np
import enum


# Define the game constants
WINDOW_SIZE = (400, 500)
BOARD_SIZE = (400, 400)
BOARD_POS = (0, 100)
TILE_FONT_SIZE = 32
TILE_FONT_COLOR = (255, 255, 255)
TILE_COLORS = {
    0: (255, 255, 255),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}
BACKGROUND_COLOR = (187, 173, 160)
GAME_OVER_COLOR = (255, 0, 0)
GAME_OVER_FONT_SIZE = 48
GAME_OVER_FONT_COLOR = (255, 255, 255)


class Direction(enum.Enum):
    UP = 1
    DOWN = 3
    LEFT = 0
    RIGHT = 2


class Game():

    # TODO: add an initial board param

    # Initialize game parameters
    def __init__(self, board_size: int=4, visualize: bool=True, command_list: list[Direction]=None):
        self.board_size = board_size
        self.tile_size = BOARD_SIZE[0] / board_size
        self.visualize = visualize
        self.command_list = command_list

        self.total_score = 0

        if not visualize:
            assert command_list is not None, 'A list of commands must be provided for headless mode'

    # Define the game logic for combining tiles
    def combine_tiles(self, arr: np.ndarray) -> Tuple[np.ndarray, int]:
        score = 0
        arr = arr[arr != 0]
        for i in range(len(arr) - 1):
            if arr[i] == arr[i+1]:
                arr[i] *= 2
                score += arr[i]
                arr[i+1] = 0
        arr = arr[arr != 0]
        arr = np.concatenate((arr, np.zeros(self.board_size - len(arr), dtype=int)))
        return arr, score

    # Define the game logic for moving tiles
    def move_tiles(self, board: np.ndarray, direction: Direction) -> Tuple[np.ndarray, int]:
        score = 0
        rotated_board = np.rot90(board, direction.value)
        for i in range(self.board_size):
            arr = rotated_board[i,:]
            arr = np.trim_zeros(arr)
            arr, row_score = self.combine_tiles(arr)
            score += row_score
            rotated_board[i,:] = arr
        new_board = np.rot90(rotated_board, -direction.value)
        return new_board, score

    # Define the game logic for adding a new tile to the board
    def add_tile(self, board: np.ndarray) -> np.ndarray:
        indices = np.where(board == 0)
        index = np.random.choice(len(indices[0]))
        x, y = indices[0][index], indices[1][index]
        board[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])
        return board
    
    # Redraw the board
    def update_screen(self, window: pygame.Surface, font: pygame.font, board: np.ndarray):
        window.fill(BACKGROUND_COLOR)
        pygame.draw.rect(window, TILE_COLORS[0], pygame.Rect(*BOARD_POS, *BOARD_SIZE))
        for i in range(self.board_size):
            for j in range(self.board_size):
                x, y = BOARD_POS[0] + j * self.tile_size, BOARD_POS[1] + i * self.tile_size
                value = board[i, j]
                color = TILE_COLORS[value]
                pygame.draw.rect(window, color, pygame.Rect(x, y, self.tile_size, self.tile_size))
                if value > 0:
                    text = font.render(str(value), True, TILE_FONT_COLOR)
                    text_rect = text.get_rect(center=(x + self.tile_size / 2, y + self.tile_size / 2))
                    window.blit(text, text_rect)
        # Update the display
        pygame.display.update()

    # Define the game logic for checking if the game is over
    def is_game_over(self, board: np.ndarray) -> bool:
        if np.count_nonzero(board) == board.size:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if i < self.board_size - 1 and board[i,j] == board[i+1,j]:
                        return False
                    if j < self.board_size - 1 and board[i,j] == board[i,j+1]:
                        return False
            return True
        return False
    
    # TODO: add functionality to save a game history (including the start val)

    # TODO: verify that random is not pseudo random

    # TODO: prepare game class for ML training
    
    def run(self):
        # Initialize the game board
        board = np.zeros((self.board_size, self.board_size), dtype=int)

        # Start game with two random tiles on the board
        board = self.add_tile(board)
        board = self.add_tile(board)
        prev_board = board.copy()

        if self.visualize:
            # Initialize Pygame
            pygame.init()
            window = pygame.display.set_mode(WINDOW_SIZE)
            pygame.display.set_caption('2048')

            # Load the font
            font = pygame.font.SysFont('Arial', TILE_FONT_SIZE, bold=True)

            self.update_screen(window, font, board)

            if self.command_list is not None:
                pygame.event.get()

        # Start the game loop
        game_over = False
        while not game_over:
            score = 0
            if self.command_list is not None:
                # Take a step through command list
                command = self.command_list.pop(0)
                board, score = self.move_tiles(board, command)
                game_over = len(self.command_list) == 0
                if self.visualize:
                    time.sleep(1)
            else:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        game_over = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_UP:
                            board, score = self.move_tiles(board, Direction.UP)
                        elif event.key == pygame.K_DOWN:
                            board, score = self.move_tiles(board, Direction.DOWN)
                        elif event.key == pygame.K_LEFT:
                            board, score = self.move_tiles(board, Direction.LEFT)
                        elif event.key == pygame.K_RIGHT:
                            board, score = self.move_tiles(board, Direction.RIGHT)

            self.total_score += score

            if not np.array_equal(prev_board, board):
                board = self.add_tile(board)

            if self.visualize and not np.array_equal(prev_board, board):
                self.update_screen(window, font, board)
                

            # Check if the game is over
            if self.is_game_over(board):
                if self.visualize:
                    game_over_text = font.render('Game Over!', True, GAME_OVER_FONT_COLOR)
                    game_over_rect = game_over_text.get_rect(center=(WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2))
                    pygame.draw.rect(window, GAME_OVER_COLOR, game_over_rect.inflate(20, 20))
                    window.blit(game_over_text, game_over_rect)

                    # Update the display
                    pygame.display.update()
                else:
                    return

            prev_board = board.copy()

        if self.visualize:
            # Quit Pygame
            pygame.quit()

if __name__ == '__main__':
    # Game(command_list=[Direction.DOWN, Direction.UP, Direction.LEFT, Direction.RIGHT]).run()    
    Game().run()    
