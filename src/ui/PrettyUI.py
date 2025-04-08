import matplotlib.pyplot as plt
from PIL import Image
import io
class PrettyUI:
    def __init__(self, board, score = 0):
        '''Create Figure and axes and update it with the givenboard and score'''
        self.emp_farb = 'xkcd:Eggshell'
        self.bg_farb = 'xkcd:greyish blue'
        self.farb_dic = {
            '': [self.emp_farb, 'Black'],
            '2':["xkcd:Light Blue",'Blue'],
            '4':["xkcd:Sky Blue",'Blue'],
            '8':["xkcd:Aqua",'Green'],
            '16':["xkcd:Pistachio", 'Brown'],
            '32':["xkcd:Grass",'White'], 
            '64':["xkcd:Puke Green",'White'],
            '128':["xkcd:Mustard Yellow",'White'],
            '256':["xkcd:GoldenRod",'White'],
            '512':["xkcd:Melon",'White'],
            '1024':["xkcd:Faded Pink",'White'],
            '2048':["xkcd:BubbleGum Pink",'White'],
            '4096':["xkcd:Lilac",'White'],
            '8192':["xkcd:Periwinkle",'White']
            }
        
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.text_list = []            
        self.fig, self.axes = plt.subplots(self.rows,self.cols, facecolor = self.bg_farb, constrained_layout = True) 
        #self.fig.set_constrained_layout_pads(top=0.7,bottom=0.11, left=0.17, right=0.855,hspace=0.2,wspace=0.2)
        self.fig.set_size_inches(self.cols+1, self.rows+1)
        self.head=self.fig.suptitle('', fontsize=30, color= 'White')
        for i in range(self.rows):
            for j in range(self.cols):
                self.a = self.axes[i,j]
                self.a.xaxis.set_visible(False)
                self.a.yaxis.set_visible(False)
                #self.a.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
                self.text_list.append(self.a.text(0.5,0.5,'', horizontalalignment='center', verticalalignment='center', fontsize =18))
                self.a.set_fc(self.emp_farb)
        plt.ion()
        plt.show()
        self.update(board,score)
          
    def update(self, new_board, new_score):
        '''Update axes with new board and score.'''
        self.board = new_board
        self.score=new_score
        self.l_places = []
        for row in range(self.rows):
            for place in self.board[row]:
                if place:
                    self.l_places.append(str(place))
                else:
                    self.l_places.append('') 
        k=0
        for i in range(self.rows):
            for j in range(self.cols):
                self.a = self.axes[i,j]
                self.num = self.l_places[k]
                                
                self.a.set_fc(self.farb_dic[self.num][0])
                self.text_list[k].set_text(self.num)
                self.text_list[k].set_color(self.farb_dic[self.num][1])
                k+=1
        
        self.score_label = '''Score: {}'''.format(self.score)
        self.head.set_text(self.score_label)

        self.fig.canvas.draw()
        plt.pause(0.1)
        
    def get_frame(self):
        """
        Capture the current plot (inclusive of axes) as a PIL Image object.
        """
        # Save the figure to a BytesIO buffer
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Open the buffer as a PIL Image
        image = Image.open(buf)
        

        return image
    
#################### Test for GUI #################
# gamestate_1 = np.array([(8192,4096,2048,1024),(64,128,256,512,),(32,16,8,4),(0,2,2,2)])
# gamestate_2 = np.array([(8192,4096,2048,1024),(64,128,256,512,),(32,16,8,4),(0,0,2,4)])
# score_1, score_2 = 11235813, 23571113

# b = Pretty_UI(gamestate_1)
# b = Pretty_UI(gamestate_1, score_1)
# b.update(gamestate_2, score_2)

# c = Shell_UI(gamestate_1, score_1)
# c.update(gamestate_2, score_2)