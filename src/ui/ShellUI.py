class ShellUI:
    def __init__(self, board, score = 0, auto_display = True):
        '''Create appropiate board string without values (assume max. number length of 6 digits)'''
        num_len=7
        num_place = '{:^'+str(num_len)+'}'
        upL, upM,upR= u'\u250F',u'\u2533',u'\u2513'
        hor, ver, emp =u'\u2501',u'\u2503',u'\u0000'
        leM, miM, riM=u'\u2523', u'\u254B', u'\u252B'
        loL,loM,loR=u'\u2517', u'\u253B', u'\u251B'
        self.score =""
        self.board = board
        self.rows = len(board)
        self.col = len(board[0])  
        self.up_row= upL+hor*num_len+(upM+hor*num_len)*(self.col-1)+upR+'\n'
        self.emp_row= (ver+num_len*emp)*self.col+ver+'\n'
        self.num_row= self.emp_row+(ver+num_place)*self.col+ver+'\n'+self.emp_row
        self.mid_row=leM+hor*num_len+(miM+hor*num_len)*(self.col-1)+riM+'\n'
        self.low_row=loL+hor*num_len+(loM+hor*num_len)*(self.col-1)+loR+'\n'
        self.board_s=self.up_row+self.num_row+(self.mid_row+self.num_row)*(self.rows-1)+self.low_row
        self.score_display = '{:^'+str((num_len+1)*self.col-1)+'}'
        if score:
            self.format_score(score)
        if auto_display:
            self.display()
    
    def format_score(self, num):
        self.score_str="Score: {}".format(num)
        self.score = self.score_display.format(self.score_str)
   
    def display(self):
        '''Take current board values as strings, format them into the empty board and print.'''
        self.l_places = []
        for row in range(len(self.board)):
            for place in self.board[row]:
                if place:
                    self.l_places.append(str(place))
                else:
                    self.l_places.append('')
        print(self.board_s.format(*self.l_places)+ self.score)
        
    def update(self,new_board, new_score = False, auto_display = True):
        '''Update board values with new list of lists in the same size as the previous one'''
        self.board = new_board
        if new_score:
            self.format_score(new_score)
        if auto_display:
            self.display()