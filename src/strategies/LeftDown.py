import random
class LeftDown:#Left then down, else right or up
    def __init__(self):
        self.counter = 0
    def __str__(self):
        return "Left Down"
    
    def move(self, game, board):#condition: game must not be over
        prio_one = ["A","S"]
        prio_two = ["D","W"]
        
        random.shuffle(prio_one)
        random.shuffle(prio_two)
        directions = prio_one + prio_two
        
        for direction in directions:
            if game.checkmove(board, direction):
                board = game.move(board, direction)
                break

        self.counter +=1
        return board[0]
    
    
"""
# Test for Left_Down_Player
player_ld = Left_Down()
spielstand = game.state()
for i in range(500):
    spielstand = game.tile_spawn(spielstand)
    if game.is_over(spielstand):
        break
    spielstand = player_ld.move(spielstand)
    print(spielstand)  
print("final gamestate: \n",spielstand, "\n Nr. of Moves: ", player_ld.counter) 
"""