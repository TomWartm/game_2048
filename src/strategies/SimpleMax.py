import numpy as np

class SimpleMax:
    def __init__(self):
        self.counter = 0
    def __str__(self):
        return "Simple Max"
    
    def move(self, game, board):
        test_board = np.copy(board)
        max_average_value = 0
        best_direction = "A"
        for direction in ["A","D","S","W"]:
            if game.future_av_value(test_board, direction) >= max_average_value and game.checkmove(test_board,direction):
                max_average_value = game.future_av_value(test_board, direction)
                best_direction = direction
        board = game.move(test_board, best_direction)
        self.counter += 1
        return board[0]
"""
# Test for Simple_max_Player
player_simple_max = Simple_Max()
spielstand = game.gamestate
for i in range(500):
    spielstand = game.tile_spawn(spielstand)
    if game.is_over(spielstand):
        break
    
    spielstand = player_simple_max.move(spielstand)
    print(spielstand)  
print("final gamestate: \n",spielstand, "\n Nr. of Moves: ", player_simple_max.counter)		
"""