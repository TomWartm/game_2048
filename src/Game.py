import random
import numpy as np

class Game:

    def __init__(self):
        """initializes the board back to empty"""
        self.gamestate=np.array([(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)])
        
    def state(self):
        """returns the current gamestate"""
        currentstate=np.copy(self.gamestate)                                    #give back a copy of the gamestate
        return currentstate
    
    def empty_spaces(self, matrix):
        """list position-tuples of all empty spaces on matrix-board"""
        empty_entries = []
        for i in range(0,4):                                                    #check all tiles
            for j in range(0,4):
                if matrix[i,j] == 0:
                    empty_entries.append((i,j))                                 #if empty (=0), add their coordinates to the list
        return empty_entries
		
    def rotate_entriesCW(self, matrix):
        """rotate a matrix clockwise"""
        use_matrix=np.copy(matrix)
        swap_matrix=np.array([(0,0,0,1),(0,0,1,0),(0,1,0,0),(1,0,0,0)])
        newmatrix=use_matrix.transpose()                                        #the composition of transposition and multiplying with a mirrored matrix "turns" the matrix
        newmatrix=np.matmul(newmatrix,swap_matrix)
        return newmatrix
		
    def rotate_entriesACW(self, matrix):
        """rotate a matrix anti-clockwise"""
        use_matrix=np.copy(matrix)
        swap_matrix=np.array([(0,0,0,1),(0,0,1,0),(0,1,0,0),(1,0,0,0)])
        newmatrix=np.matmul(use_matrix,swap_matrix)
        newmatrix=newmatrix.transpose()
        return newmatrix
	
    def is_over(self, matrix):
        """checks if legal move on the board"""
        returnvalue = True
        if not len(self.empty_spaces(matrix)) == 0:                              #if there are empty tiles on the board there must be a legal move
            returnvalue= False
        else:
            i=0
            while i in range(0,4) and returnvalue==True:
                j=0
                while j in range(0,3) and returnvalue==True:
                    if matrix[i,j]== matrix[i,j+1]:                             #check vertically if there are adjacent cells with the same value that could be combined which would imply a legal move
                        returnvalue= False
                    j+=1
                i+=1
            i=0
            while i in range(0,3) and returnvalue == True:
                j=0
                while j in range(0,4) and returnvalue == True:
                    if matrix[i,j]== matrix[i+1,j]:                              #same check horizontally
                        returnvalue= False
                    j+=1
                i+=1
        return returnvalue
        
    def checkmoveUP(self, matrix):
        """checks if the given boardstate is compatible with the move UP/N"""
        legality = False
        j=0
        while j in range(0,4) and legality==False:
            i=0
            while i in range(0,3) and legality==False:
                if matrix[i,j] == 0 and not matrix[i+1,j]  == 0:                #checks if there is an empty tile directly above a nonempty one, if so the move up must be legal
                    legality = True
                i+=1
            j+=1
        j=0
        while j in range(0,4) and legality == False:
            i=0
            while i in range(0,3) and legality == False:
                if matrix[i,j] == matrix[i+1,j] and not matrix[i,j] == 0:       #checks if there are two identical (nonempty) tiles directly above eachother, is so a move combining them must be legal
                    legality = True
                i+=1
            j+=1
        return legality
	
    def checkmove(self, matrix, direction):
        """checks if a boardstate is compatible with a move in the given direction"""
        test_matrix = np.copy(matrix)
        movelegality = False
        if direction == "W":                                                    #this applies checkmoveUp by just rotating the matrix into an orientation where our move becomes N/UP
            movelegality= self.checkmoveUP(test_matrix)
        if direction == "D":
            test_matrix = self.rotate_entriesACW(test_matrix)
            movelegality= self.checkmoveUP(test_matrix)
        if direction == "A":
            test_matrix = self.rotate_entriesCW(test_matrix)
            movelegality= self.checkmoveUP(test_matrix)
        if direction == "S":
            test_matrix = self.rotate_entriesCW(test_matrix)
            test_matrix = self.rotate_entriesCW(test_matrix)
            movelegality= self.checkmoveUP(test_matrix)
        return movelegality
    
    def current_score(self, matrix):                                                  #simply returns the total value of a board by adding all tile values.
        """returns the score of a matrix-board"""
        score=0
        for i in range(0,4):
            for j in range(0,4):
                score+=matrix[i,j]
        return score
    
    def highest_tile(self, matrix):
        """ returns the highest tile of a matrix board"""
        tile_num = 0
        for i in range(0,4):
            for j in range(0,4):
                if matrix[i,j] > tile_num:
                    tile_num = matrix[i,j]
        return tile_num
    
    def average_value(self, matrix):                                                  #this is a more advanced score version where we divide the score by the number of nonempty tiles
        """returns the average tilevalue of a board 
        (this excludes empty tiles)"""
        emptylist=self.empty_spaces(matrix)
        totalvalue=self.current_score(matrix)
        averagevalue=totalvalue/(16-len(emptylist))
        return averagevalue
	
    def moveUP(self, matrix):
        """executes the move UP/N on a boardstate"""
        newmatrix=np.copy(matrix)
        move_matrix=np.copy(matrix)
        for j in range(0,4):                                                    #consolidates the tiles at the top edge
            nonzero_tiles=[]
            for i in range(0,4):
                if not move_matrix[i,j] == 0:
                    nonzero_tiles.append(move_matrix[i,j])
            for k in range(0,4):
                if k < len(nonzero_tiles):
                    newmatrix[k,j]=nonzero_tiles[k]
                else:
                    newmatrix[k,j]=0
        final_matrix = np.copy(newmatrix)
        for j in range(0,4):                                                    #merges duplicate neighbors into superior tiles and ensure the board gets adjusted
            if newmatrix[0,j]==newmatrix[1,j]:                                  #checks if the first and second entry are identical
                final_matrix[0,j]= newmatrix[0,j]+newmatrix[1,j]
                if newmatrix[2,j]==newmatrix[3,j]:                              #this is the case where 1st=2nd and 3rd=4th so we have the combined tiles in the 1st and 2nd entry and then fill with 0s
                    final_matrix[1,j]=newmatrix[2,j]+newmatrix[3,j]
                    final_matrix[2,j]=0
                    final_matrix[3,j]=0
                else:                                                           #in this case only 1st=2nd so we merge and then move up the other 2 tiles and add a 0
                    final_matrix[1,j]=newmatrix[2,j]
                    final_matrix[2,j]=newmatrix[3,j]
                    final_matrix[3,j]=0
            elif newmatrix[1,j]==newmatrix[2,j]:                                #this case is only 2nd=3rd so 1st entry stays unchanged 
                final_matrix[1,j]=newmatrix[1,j]+newmatrix[2,j]
                final_matrix[2,j]=newmatrix[3,j]
                final_matrix[3,j]=0
            elif newmatrix[2,j]==newmatrix[3,j]:                                #last "special case" where 3rd=4th so they merge and we empty the last cell.
                final_matrix[2,j]=newmatrix[2,j]+newmatrix[3,j]
                final_matrix[3,j]=0
        return final_matrix
    
    def move(self, matrix, direction):                                                 #applies moveUP by rotating the matrix into the needed position and then afterwards rotates them back
        """checks if a move is legal on a boardstate and executes it if legal
        if move legal returns the new boardstate and True
        if move illegal returns the original boardstate and False"""
        legality=False
        move_matrix=np.copy(matrix)
        return_matrix=np.copy(matrix)
        manipulated_matrix = np.copy(matrix)
        legality= self.checkmove(move_matrix,direction)
        if direction== "W" and legality == True:
            return_matrix = self.moveUP(move_matrix)
        if direction== "D" and legality == True:
            manipulated_matrix = self.rotate_entriesACW(move_matrix)
            manipulated_matrix = self.moveUP(manipulated_matrix)
            return_matrix = self.rotate_entriesCW(manipulated_matrix)
        if direction== "A" and legality == True:
            manipulated_matrix = self.rotate_entriesCW(move_matrix)
            manipulated_matrix = self.moveUP(manipulated_matrix)
            return_matrix = self.rotate_entriesACW(manipulated_matrix)
        if direction == "S" and legality == True:
            manipulated_matrix = self.rotate_entriesACW(move_matrix)
            manipulated_matrix = self.rotate_entriesACW(manipulated_matrix)
            manipulated_matrix = self.moveUP(manipulated_matrix)
            manipulated_matrix = self.rotate_entriesACW(manipulated_matrix)
            return_matrix = self.rotate_entriesACW(manipulated_matrix)
        return (return_matrix,legality)
    
    def future_av_value(self, matrix, direction):                                      #hypothetically executes a move and gives back the average (nonempty) tilevalue after that move
        """predict the average tilevalue after a given move on the current board
        this does not take into account the random tilespawn
        returns 0 if move illegal"""
        legality=False
        future_matrix=np.copy(matrix)
        predict_matrix=np.copy(matrix)
        future_value=0
        future_matrix,legality= self.move(predict_matrix,direction)
        if legality == True:
            future_value= self.average_value(future_matrix)
            return future_value
        else:
            return 0
    
    def tile_spawn(self, matrix):                                                     #this first picks a random empty tile using the empty_spaces and then inserts a 2 or 4 into that position 
        """spawns a 2 or a 4 on the board in an empty spot
        90% chance for 2
        10% for 4"""
        spawned_matrix = np.copy(matrix)
        empty_spaces = self.empty_spaces(spawned_matrix)
        empty_count = len(empty_spaces)
        spawnposition=random.randint(1,empty_count) -1
        a,b=empty_spaces[spawnposition]
        two_or_four=random.randint(1,10)
        if two_or_four== 1:
            spawned_matrix[a,b]=4
        else:
            spawned_matrix[a,b]=2
        return spawned_matrix