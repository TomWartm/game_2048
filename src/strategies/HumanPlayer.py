class HumanPlayer:
    
    def __init__(self, name):
        self.counter = 0
        self.name = name
    
    def move(self, game, board): #condition: game must not be over
        movelegality = False
        direction = None
        while not movelegality :
            direction = str(input("Please Enter a valid Direction from W,A,S,D: ")).upper()
            movelegality = game.checkmove(board,direction)
            if direction == "EXIT":
                movelegality = 1
        if direction == "EXIT":
            return direction
        else:
            board = game.move(board,direction)
            self.counter+=1
            return board[0]

    def __str__(self):
        return self.name        
    