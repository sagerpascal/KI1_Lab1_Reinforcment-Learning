import numpy as np
from game2048 import Game2048Env
from rl.core import Processor


class InputProcessor(Processor):
    """
    Der weiter oben verwendete `InputProcessor` erweitert die `Processor` des `rl.core` und ist eine Art Pre-Prozessor
     für den Input in das neurale Netzwerk. Der `InputProcessor` berechnet alle möglichen Kombinationen für die
     nächsten zwei Schritte, welche durch den Spieler (Schieben in die Richtungen `UP`, `DOWN`, `LEFT`, `RIGHT`)
     und den Zufallsgenerator des Spiels (neue Zahl) vorgenommen werden können. Dies wird durch eine `One-Hot-Encoding`
     Methode gemacht, wobei für jede mögliche Zahl 2^n mit n=0, 1, ..., 16  eine Matrix erstellt wird und eine 0 anzeigt,
      dass an entsprechender Stelle die Zahl "2^n nicht vorkommt und eine 1, dass die Zahl 2^n an entsprechender Stelle
      vorhanden ist.
    """

    def __init__(self, num_one_hot_matrices=16):
        """
        num_one_hot_matrices: Anzahl Matrizen zum Encoden jedes Games (2^16 ist 65536 als höchste Tilde)
        """
        self.num_one_hot_matrices = num_one_hot_matrices
        self.window_length = 1
        self.model = 'dnn'
        self.game_env = Game2048Env()

        # Dict mit 2^n:n für alle möglichen Zahlen -> {2: 1, 4: 2, 8: 3, ..., 16384: 14, 32768: 15, 65536: 16}
        self.table = {2 ** i: i for i in range(1, self.num_one_hot_matrices)}
        self.table[0] = 0

    def one_hot_encoding(self, grid):
        """
        Konvertierung in one_hot_encoding (siehe Blogpost für genauere Beschreibung)
        """
        grid_one_hot = np.zeros(shape=(self.num_one_hot_matrices, 4, 4))
        for i in range(4):
            for j in range(4):
                element = grid[i, j]
                grid_one_hot[self.table[element], i, j] = 1
        return grid_one_hot

    def get_grids_after_move_of_player(self, grid):
        """
        Gibt das Board nach dem nächsten Move zurück

        grid = aktuelles Board
        :return Liste mit allen 4 möglichen neuen Boards
        """
        grids_list = []
        for move in range(4):
            grid_orig = grid.copy()
            self.game_env.set_board(grid_orig)
            try:
                _ = self.game_env.move(move)
            except:
                pass
            grid_after = self.game_env.get_board()
            grids_list.append(grid_after)
        return grids_list

    def process_observation(self, observation):
        """
        Wird durch Keras-RL aufgerufen, um für jede Observation alle Möglichkeiten in den Nachfolgenden 2 Schritten
        zu berechnen (zurückgegeben in One-Hot-Encoding), bevor diese in das neurale Netzwerk weitergeleitet werden

        :return: Liste mit allen möglichen Grids in den nächsten zwei Schritten, in Form der One-Hot-Encoding

        """
        # Sicherstellen, dass 4x4 Matrix
        observation = np.reshape(observation, (4, 4))

        grids_after_player = self.get_grids_after_move_of_player(observation)
        grids_after_chance = []  # nach Zufallsgenerator des Spiels
        for grid in grids_after_player:
            grids_after_chance.append(grid)
            grids_temp = self.get_grids_after_move_of_player(grid)
            for grid_temp in grids_temp:
                grids_after_chance.append(grid_temp)
        grids_list = np.array([self.one_hot_encoding(grid) for grid in grids_after_chance])
        return grids_list

    def process_state_batch(self, batch):
        pass
