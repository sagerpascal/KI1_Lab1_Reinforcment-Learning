import os, fnmatch, pickle

import numpy as np
import random

from game2048 import Game2048Env
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import ModelIntervalCheckpoint

from callbacks import TrainEpisodeLogger2048, KerasCallbackLogger
from processors import InputProcessor

projekt_pfad = 'D://Projekte/DQN2048'
daten_pfad = projekt_pfad + '/daten'
if not os.path.exists(daten_pfad):
    os.makedirs(daten_pfad)


NUM_ACTIONS_PRO_AGENT = 4  # Anzahl Agent Actions -> UP, DOWN, LEFT, RIGHT => 4
STATES = 1  # 1 State, da fully observable
INPUT_MATRIX_SIZE = (4, 4)  # Input in Neurales Netzwerk -> 4x4 Matrix

MEMORY_SIZE = 6000
TARGET_MODEL_UPDATE = 1000

NUM_STEPS = 1000000  # Anzahl Steps für Training
NUM_STEPS_ANNEALED = 80000  # Anzahl Schritte in LinearAnnealedPolicy
NUM_STEPS_WARMUP = 5000  # Anzahl Schritte um Speicher zu füllen vor Training

# One Hot Encoding nötig, damit Daten korrekt interpretiert werden (es gilt nicht 0<1<2)
NUM_MATRIX = 16  # Anzahl Matrix für Encoding des Spielfelds

ENVIRONMENT_NAME = '2048'
environment = Game2048Env()

# Seed, damit reproduzierbar
random_seed = 123
random.seed(random_seed)
np.random.seed(random_seed)
environment.seed(random_seed)

# Versuche bestehendes Modell zu laden
try:
    model_name = fnmatch.filter(os.listdir(daten_pfad), '*_model.h5')[-1]
    nb_training_steps = int(model_name.split("_")[4])
    NUM_STEPS = NUM_STEPS - nb_training_steps  # Anz. Verbleibende Steps
    model_path = daten_pfad + '/' + model_name
    model = load_model(model_path)

# Erstelle Modell, wenn keines gefunden wurde
except:
    processor = InputProcessor(num_one_hot_matrices=NUM_MATRIX)
    INPUT_SHAPE = (STATES, 4 + 4 * 4, NUM_MATRIX,) + INPUT_MATRIX_SIZE  # Beispiel wäre (1,20,16,4,4); 4+4*4 ist die Anzahl Gruds in den nächsten 2 Steps

    NUM_DENSE_NEURONS_L1 = 1024  # Anzahl Neuronen im 1. Layer
    NUM_DENSE_NEURONS_L2 = 512  # Anzahl Neuronen im 2. Layer
    NUM_DENSE_NEURONS_L3 = 256  # Anzahl Neuronen im 3. Layer
    ACTIVATION_FTN = 'relu'
    ACTIVATION_FTN_OUTPUT = 'linear'

    # DNN model
    model = Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(Dense(units=NUM_DENSE_NEURONS_L1, activation=ACTIVATION_FTN))
    model.add(Dense(units=NUM_DENSE_NEURONS_L2, activation=ACTIVATION_FTN))
    model.add(Dense(units=NUM_DENSE_NEURONS_L3, activation=ACTIVATION_FTN))
    model.add(Dense(units=NUM_ACTIONS_PRO_AGENT, activation=ACTIVATION_FTN_OUTPUT))
    print(model.summary())


# Versuche bestehendes Training zu laden
try:
    agent_memory_name = fnmatch.filter(os.listdir(daten_pfad), '*_agentmem.pkl')[-1]
    nb_training_steps = int(agent_memory_name.split("_")[0])
    NUM_STEPS = NUM_STEPS - nb_training_steps  # Anz. Verbleibende Steps
    pickle_filepath = daten_pfad + '/' + agent_memory_name
    (memory, memory.actions, memory.rewards, memory.terminals, memory.observations) = pickle.load(open(pickle_filepath, "rb"))

# sonst neuen Speicher erstellen
except:
    memory = SequentialMemory(limit=MEMORY_SIZE, window_length=STATES)


# Versuche Agent zu laden
try:
    agent_ame = fnmatch.filter(os.listdir(daten_pfad), '*_agent.pkl')[-1]
    nb_training_steps = int(agent_ame.split("_")[0])
    NUM_STEPS = NUM_STEPS - nb_training_steps # Anz. Verbleibende Steps
    agent_filepath = daten_pfad + '/' + agent_ame
    dqn = pickle.load(open(agent_filepath, "rb"))

# sonst erstelle neuen Agent
except:
    TRAIN_POLICY = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=0.05, value_min=0.05, value_test=0.01, nb_steps=NUM_STEPS_ANNEALED)
    TEST_POLICY = EpsGreedyQPolicy(eps=.01)
    dqn = DQNAgent(model=model, nb_actions=NUM_ACTIONS_PRO_AGENT, test_policy=TEST_POLICY, policy=TRAIN_POLICY,
                   memory=memory, processor=processor, nb_steps_warmup=NUM_STEPS_WARMUP, gamma=.99, target_model_update=TARGET_MODEL_UPDATE,
                   train_interval=4, delta_clip=1.)  # , batch_size=BATCH_SIZE)

    dqn.compile(Adam(lr=.00025), metrics=['mse'])



# TRAINING

weights_filepath = daten_pfad + '/weights.h5f'
checkpoint_weights_filepath = daten_pfad + '/weights_{step}.h5f'
csv_filepath = daten_pfad + '/train.csv'

# Setze Callbacks von Keros-RL
_callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filepath, interval=250000)]
_callbacks += [TrainEpisodeLogger2048(csv_filepath)]

# Trainiere
dqn.fit(environment, callbacks=_callbacks, nb_steps=NUM_STEPS, visualize=False, verbose=0) # Starte Training
dqn.save_weights(weights_filepath, overwrite=True) # Speichere die Gewichte nach dem Training
memory = (memory, memory.actions, memory.rewards, memory.terminals, memory.observations) # Aktualisiere memory vor dem Speichern

# Speichere in bestehende File
if 'nb_training_steps' in locals():
    agentmem_filepath = daten_pfad + '/{}_agentmem.pkl'.format(NUM_STEPS + nb_training_steps)
    pickle.dump(memory, open(agentmem_filepath, "wb"))
    model_path = daten_pfad + '/{}_model.h5'.format(NUM_STEPS + nb_training_steps)
    model.save(model_path)

# Speichere in neue Files
else:
    agentmem_filepath = daten_pfad + '/{}_agentmem.pkl'.format(NUM_STEPS)
    pickle.dump(memory, open(agentmem_filepath, "wb"), protocol=-1)

    agent_filepath = daten_pfad + '/{}_agent.pkl'.format(NUM_STEPS)
    pickle.dump(dqn, open(agent_filepath, "wb"), protocol=-1)

    model_path = daten_pfad + '/{}_model.h5'.format(NUM_STEPS)
    model.save(model_path)  # creates a HDF5 file 'my_model.h5'


# "Abschlussarbeiten"
environment.reset()
_callbacks = [KerasCallbackLogger()]
dqn.test(environment, nb_episodes=5, visualize=False, verbose=0, callbacks=_callbacks)

