# Keras-RL ruft diese Funktionen auf nach jeder Episode (Calbacks)
#

from os.path import exists
import csv
import numpy as np
import matplotlib.pyplot as plt
from rl.callbacks import Callback, TestLogger


class KerasCallbackLogger(TestLogger):
    def on_episode_end(self, episode, logs):
        # Callback von Keras-RL nach jeder Episode

        grid = self.env.get_board()
        print('Episode: ' + str(episode + 1) + 'Max Tile: ' + str(np.amax(grid)) + ' Punkte Episode: ' + str(
            logs['episode_reward']) + 'Steps: ' + str(logs['nb_steps']))
        print("Grid am Ende: \n{0}\n".format(grid))


class TrainEpisodeLogger2048(Callback):
    # Plote die Grafiken und schreibe CSV-File

    def __init__(self, filePath):
        self.observations = {}
        self.rewards = {}
        self.max_tile = {}
        self.step = 0
        self.episodes = []
        self.max_tiles = []
        self.episodes_rewards = []
        self.fig_max_tile = plt.figure()
        self.ax1 = self.fig_max_tile.add_subplot(1, 1, 1)
        self.fig_reward = plt.figure()
        self.ax2 = self.fig_reward.add_subplot(1, 1, 1)


        self.max_tiles_means = 0
        self.episodes_rewards_means = 0
        self.fig_max_tile_mean = plt.figure()
        self.ax3 = self.fig_max_tile_mean.add_subplot(1, 1, 1)
        self.fig_reward_mean = plt.figure()
        self.ax4 = self.fig_reward_mean.add_subplot(1, 1, 1)

        self.nb_episodes_for_mean = 50
        self.episode_counter = 0

        # CSV file:
        if exists(filePath):
            csv_file = open(filePath, "a")  # a = append
            self.csv_writer = csv.writer(csv_file, delimiter=',')
        else:
            csv_file = open(filePath, "w")  # w = write (clear and restart)
            self.csv_writer = csv.writer(csv_file, delimiter=',')
            headers = ['episode', 'episode_steps', 'episode_reward', 'max_tile']
            self.csv_writer.writerow(headers)

    def on_episode_begin(self, episode, logs):
        # Werte rücksetzen (Aufgerufen von Keras-RL)
        self.observations[episode] = []
        self.rewards[episode] = []
        self.max_tile[episode] = 0

    def on_episode_end(self, episode, logs):
        # Daten ausgeben und CSV schreiben (Aufgerufen von Keras-RL)
        self.episode_counter += 1
        self.episodes = np.append(self.episodes, episode + 1)
        self.max_tiles = np.append(self.max_tiles, self.max_tile[episode])
        self.episodes_rewards = np.append(self.episodes_rewards, np.sum(self.rewards[episode]))

        print('Episode: ' + str(episode + 1) + 'Episode Steps: ' +
              str(len(self.observations[episode])) + 'Max Tile: ' + str(self.max_tiles[-1]) + ' Punkte Episode: ' + str(
            self.episodes_rewards[-1]))

        # Speichere CSV:
        self.csv_writer.writerow(
            (episode + 1, len(self.observations[episode]), self.episodes_rewards[-1], self.max_tiles[-1]))

        # Plots erstellen -> kopiert
        if self.episode_counter % self.nb_episodes_for_mean == 0:
            self.max_tiles_means = np.append(self.max_tiles_means, np.mean(self.max_tiles[-self.nb_episodes_for_mean:]))
            self.fig_max_tile_mean.clear()
            plt.figure(self.fig_max_tile_mean.number)
            plt.plot(np.arange(0, self.episode_counter + self.nb_episodes_for_mean, self.nb_episodes_for_mean), self.max_tiles_means)
            plt.title("Höchster Block (in den letzten {} Episoden)".format(self.nb_episodes_for_mean))
            plt.xlabel("Episode")
            plt.ylabel("Durchschn. höchster Block")
            plt.pause(0.01)

            self.episodes_rewards_means = np.append(self.episodes_rewards_means, np.mean(self.episodes_rewards[-self.nb_episodes_for_mean:]))
            self.fig_reward_mean.clear()
            plt.figure(self.fig_reward_mean.number)
            plt.plot(np.arange(0, self.episode_counter + self.nb_episodes_for_mean, self.nb_episodes_for_mean), self.episodes_rewards_means)
            plt.title("Punkte-Durchschnitt (in den letzten {} Episoden)".format(self.nb_episodes_for_mean))
            plt.xlabel("Episode")
            plt.ylabel("Punkte-Durchschnitt")
            plt.pause(0.01)

        # Figures: Points
        self.fig_max_tile.clear()
        plt.figure(self.fig_max_tile.number)
        plt.scatter(self.episodes, self.max_tiles, s=1)
        plt.title("Höchster Block pro Episode")
        plt.xlabel("Episode")
        plt.ylabel("Höchster Block")
        plt.pause(0.01)

        self.fig_reward.clear()
        plt.figure(self.fig_reward.number)
        plt.scatter(self.episodes, self.episodes_rewards, s=1)
        plt.title("Punkte pro Episode")
        plt.xlabel("Episode")
        plt.ylabel("Punkte")
        plt.pause(0.01)

        # Resourcen freigeben
        del self.observations[episode]
        del self.rewards[episode]
        del self.max_tile[episode]

    def on_step_end(self, step, logs):
        # Update der Statistiken
        episode = logs['episode']
        self.observations[episode].append(logs['observation'])
        self.rewards[episode].append(logs['reward'])
        self.max_tile[episode] = logs['info']['max_tile']
        self.step += 1
