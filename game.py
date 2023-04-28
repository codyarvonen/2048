import pygame
import numpy as np
import enum

class Direction(enum.Enum):
    UP = 1
    DOWN = 3
    LEFT = 0
    RIGHT = 2

# Define the game constants
WINDOW_SIZE = (400, 500)
BOARD_SIZE = (400, 400)
BOARD_POS = (0, 100)
TILE_SIZE = 100
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


# Define the game logic for combining tiles
def combine_tiles(arr: np.ndarray) -> np.ndarray:
    arr = arr[arr != 0]
    for i in range(len(arr) - 1):
        if arr[i] == arr[i+1]:
            arr[i] *= 2
            arr[i+1] = 0
    arr = arr[arr != 0]
    arr = np.concatenate((arr, np.zeros(4 - len(arr), dtype=int)))
    return arr

# Define the game logic for moving tiles
def move_tiles(board: np.ndarray, direction: Direction) -> np.ndarray:
    rotated_board = np.rot90(board, direction.value)
    for i in range(4):
        arr = rotated_board[i,:]
        arr = np.trim_zeros(arr)
        arr = combine_tiles(arr)
        rotated_board[i,:] = arr
    new_board = np.rot90(rotated_board, -direction.value)
    return new_board

# Define the game logic for adding a new tile to the board
def add_tile(board: np.ndarray) -> np.ndarray:
    indices = np.where(board == 0)
    index = np.random.choice(len(indices[0]))
    x, y = indices[0][index], indices[1][index]
    board[x, y] = np.random.choice([2, 4], p=[0.9, 0.1])
    return board

# Define the game logic for checking if the game is over
def is_game_over(board: np.ndarray) -> bool:
    if np.count_nonzero(board) == board.size:
        for i in range(4):
            for j in range(4):
                if i < 3 and board[i,j] == board[i+1,j]:
                    return False
                if j < 3 and board[i,j] == board[i,j+1]:
                    return False
        return True
    return False

if __name__ == '__main__':

    # Initialize the game board
    board = np.zeros((4, 4), dtype=int)

    # Initialize Pygame
    pygame.init()
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('2048')

    # Load the font
    font = pygame.font.SysFont('Arial', TILE_FONT_SIZE, bold=True)

    # Start the game loop
    game_over = False
    while not game_over:

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    board = move_tiles(board, Direction.UP)
                    board = add_tile(board)
                elif event.key == pygame.K_DOWN:
                    board = move_tiles(board, Direction.DOWN)
                    board = add_tile(board)
                elif event.key == pygame.K_LEFT:
                    board = move_tiles(board, Direction.LEFT)
                    board = add_tile(board)
                elif event.key == pygame.K_RIGHT:
                    board = move_tiles(board, Direction.RIGHT)
                    board = add_tile(board)

        # Draw the board
        window.fill(BACKGROUND_COLOR)
        pygame.draw.rect(window, TILE_COLORS[0], pygame.Rect(*BOARD_POS, *BOARD_SIZE))
        for i in range(4):
            for j in range(4):
                x, y = BOARD_POS[0] + j * TILE_SIZE, BOARD_POS[1] + i * TILE_SIZE
                value = board[i, j]
                color = TILE_COLORS[value]
                pygame.draw.rect(window, color, pygame.Rect(x, y, TILE_SIZE, TILE_SIZE))
                if value > 0:
                    text = font.render(str(value), True, TILE_FONT_COLOR)
                    text_rect = text.get_rect(center=(x + TILE_SIZE / 2, y + TILE_SIZE / 2))
                    window.blit(text, text_rect)

        # Check if the game is over
        if is_game_over(board):
            game_over_text = font.render('Game Over!', True, GAME_OVER_FONT_COLOR)
            game_over_rect = game_over_text.get_rect(center=(WINDOW_SIZE[0] / 2, WINDOW_SIZE[1] / 2))
            pygame.draw.rect(window, GAME_OVER_COLOR, game_over_rect.inflate(20, 20))
            window.blit(game_over_text, game_over_rect)

        # Update the display
        pygame.display.update()

    # Quit Pygame
    pygame.quit()
