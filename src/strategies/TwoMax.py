import numpy as np

class TwoMax:
    def __init__(self):
        self.counter = 0
        
    def __str__(self):
        return "Two Max"
    
    def move(self, game, board):
        
        max_sum_of_max_av_value = 0
        best_direction = "A"
        for first_direction in ["A","D","S","W"]: # go through all directions 
            sum_of_max_av_values = 0
            first_test_board = np.copy(board)
            if game.checkmove(first_test_board,first_direction):
                first_test_board = game.move(first_test_board,first_direction)
                first_test_board = game.tile_spawn(first_test_board[0]) 
                
                for seccond_direction in ["A","D","S","W"]: # sum the possible values from all directions for the seccond time 
                    sum_of_max_av_values += game.future_av_value(first_test_board, seccond_direction)
                if sum_of_max_av_values >= max_sum_of_max_av_value:
                    max_sum_of_max_av_value = sum_of_max_av_values
                    best_direction = first_direction # and choose the first direction with the highest sum
        board = game.move(board,best_direction)
        self.counter += 1
        return board[0]
        
      
""" 
# Test for Prob_Max
player_prob_max = Prob_Max()
spielstand = game.gamestate
for i in range(1000):
    spielstand = game.tile_spawn(spielstand)
    if game.is_over(spielstand):
        break
    spielstand = player_prob_max.move(spielstand)
     
print("final gamestate: \n",spielstand, "\n Nr. of Moves: ", player_prob_max.counter,"\n final av. Value: ", game.average_value(spielstand) )	   
"""