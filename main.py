import numpy as np

import csv

import matplotlib
from tqdm import tqdm
from src.Game import Game
from src.strategies.HumanPlayer import HumanPlayer
from src.strategies.LeftDown import LeftDown
from src.strategies.Random import Random
from src.strategies.TwoMax import TwoMax
from src.strategies.SimpleMax import SimpleMax
from src.strategies.QLearning import QLearning
from src.ui.PrettyUI import PrettyUI
from src.ui.ShellUI import ShellUI


matplotlib.use('TkAgg')
                        

	
if __name__ == "__main__":
    # Main program to execute the game
    game = Game()
    #Choose if you want to play by yourself or let the computer play the game
    player_modes=["VISUAL", "SHELL","STATISTICAL", "STA"]
    player_mode=None
    while player_mode not in player_modes:
        player_mode=input('- To play the game by yourself enter: "visual" \n\
    - To play the game by yourself using just the shell enter: "shell" \n\
    - To run a computer simulation enter: "statistical" \n\
    - To exit the program enter: "Exit"  ').upper()
        # Quits program if user enters exit
        if player_mode == "EXIT":
                print("You have left the game!")
                break
            
    # Execute the mode where a human player plays the game
    player= None
    if player_mode =="SHELL":
        player = HumanPlayer("name")
        spielstand = game.gamestate
        spielstand= game.tile_spawn(spielstand)
        spielstand= game.tile_spawn(spielstand)
        gui = ShellUI(spielstand)
        while not game.is_over(spielstand):
            print("Current score: ", game.current_score(spielstand), "\nNumber of moves ", player.counter, "\nHighest tile number: ", game.highest_tile(spielstand))
            test_spielstand = player.move(game, spielstand)

            spielstand = np.copy(test_spielstand)
            spielstand = game.tile_spawn(spielstand)
            gui.update(spielstand)
        print("Game is over! \nYour score is: ", game.current_score(spielstand), "\nNumber of moves ", player.counter, "\nHighest tile number: ", game.highest_tile(spielstand))

    elif player_mode =="VISUAL":
        player = HumanPlayer("name")
        spielstand = game.gamestate
        spielstand= game.tile_spawn(spielstand)
        spielstand= game.tile_spawn(spielstand)
        gui = PrettyUI(spielstand)
        while not game.is_over(spielstand):
            print("\nNumber of moves ", player.counter, "\nHighest tile number: ", game.highest_tile(spielstand))
            test_spielstand = player.move(game, spielstand)

            spielstand = np.copy(test_spielstand)
            spielstand = game.tile_spawn(spielstand)
            gui.update(spielstand, game.current_score(spielstand))
        print("Game is over! \nYour score is: ", game.current_score(spielstand), "\nNumber of moves ", player.counter, "\nHighest tile number: ", game.highest_tile(spielstand))

    # Execute statistical mode
    else:   
        players=[QLearning()]
        print("Please wait for the data to be loaded...")
        with open('data/statistical.csv', 'w', newline='') as f:
            thewriter = csv.writer(f)
            thewriter.writerow(['Strategy','Score', 'Number of moves','Highest Tile','Game over', 'Iteration'])
            for k in tqdm(range(100000)): # execute game k times per strategy
                for player in players: # iterate over different computer strategies
                    spielstand = game.gamestate
                    player.counter = 0
                    for i in range(2000):
                        spielstand = game.tile_spawn(spielstand)
                        if game.is_over(spielstand):
                            thewriter.writerow([player, game.current_score(spielstand), player.counter, game.highest_tile(spielstand), game.is_over(spielstand), k])
                            break
                        spielstand = player.move(game, spielstand)
                        
                        if k % 1000 == 0:
                            thewriter.writerow([player, game.current_score(spielstand), player.counter, game.highest_tile(spielstand), game.is_over(spielstand), k])

            print('Computer simulation finished. Filename: statistical.csv ' )
                        
