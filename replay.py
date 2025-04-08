from src.Game import Game
from src.ui.PrettyUI import PrettyUI
import pandas as pd
import numpy as np
import time
import ast


if __name__ == "__main__":
    # Replay a played game in visual mode
    df = pd.read_csv('data/statistical.csv')
    df['Board'] = df['Board'].apply(lambda x: np.array(ast.literal_eval(x)))

    game = Game()
    initial_board = np.zeros((4, 4), dtype=int)  # Initialize an empty 4x4 board
    gui = PrettyUI(initial_board)

    max_score = df['Score'].max()
    iteration = df[df['Score'] == max_score]['Iteration'].values[0]

    df = df[df['Iteration'] == 0]
    strategy = df['Strategy'].values[0]
    frames = []
    for index, row in df.iterrows():
        board = row['Board']
        score = row['Score']
        gui.update(board, score)

        # Capture the current figure as a PIL Image
        frame = gui.get_frame()
        frames.append(frame)
        
        time.sleep(0.1)


    # Save the frames as a GIF
    frames[0].save(f'img/replay_{strategy}_23672.gif', save_all=True, append_images=frames[1:], duration=1000, loop=0)
