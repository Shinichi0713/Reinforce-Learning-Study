from ple.games.catcher import Catcher
from ple.ple import PLE

game = Catcher()
env = PLE(game, fps=30, display_screen=True)
env.init()