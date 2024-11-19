import json
from agent import Agent
from game import Snake
from settings import *


class Game:
    def __init__(self):
        with open('par_lev.json', 'r') as file:
            self.json_pars = json.load(file)

        self.pars = self.json_pars.get("game_pars")
        self.train(self.pars)

    # Save run stats to a txt file at a specified path
    def save_to_file(self, path, game_num, score, record):
        """
        Save the game, score, and record to a txt file.
        """
        with open(path, "a+") as file:
            file.write(f"{game_num} {score} {record}\n")

    def train(self, pars):
        """
        Train game and run each step as
        a sequence of frames.
        """
        # Initialize
        record = 0
        game = Snake()
        agent = Agent(input_channels=1, output_size=OUTPUT_SIZE, input_height=10, input_width=10, pars=pars, show_plot_num=50)

        while True:
            # Get game state (replace OpenCV logic with basic state extraction)
            state_old = game.cnn_input()

            # Decide action based on state
            final_move = agent.get_action(state_old)

            # Move the snake
            reward, done, score = game.play_step(action=final_move, kwargs=pars)

            # Get the new state
            state_new = game.cnn_input()

            # Train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # Remember
            agent.remember(state_old, final_move, reward, state_new, done)

            # End game condition
            if pars.get('num_games', DEFAULT_END_GAME_POINT) != -1:
                if agent.n_games > pars.get('num_games', DEFAULT_END_GAME_POINT):
                    break

            # When the game is over
            if done:
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                # Update high score
                if score > record:
                    record = score
                    agent.model_cnn.save()

                # Print game information
                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                # Save game information
                self.save_to_file(f"./{pars.get('graph', 'test')}.txt", agent.n_games, score, record)


if __name__ == "__main__":
    Game()
