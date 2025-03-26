import random

class Random:
    def __init__(self):
        self.counter = 0
    def __str__(self):
        return "Random"
    
    def move(self, game, board):
        direction = random.choice(["W","D","S","A"])
        while not game.checkmove(board, direction):
            direction = random.choice(["W","D","S","A"])
        board = game.move(board, direction)
        self.counter += 1
        return board[0]
"""
# Test for Random_Player
player_rand = Random()
spielstand = game.gamestate
for i in range(500):
    spielstand = game.tile_spawn(spielstand)
    if game.is_over(spielstand):
        break
    spielstand = player_rand.move(spielstand)
    print(spielstand)  
print("final gamestate: \n",spielstand, "\n Nr. of Moves: ", player_rand.counter)
"""		