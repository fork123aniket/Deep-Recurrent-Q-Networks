import numpy as np
import tensorflow as tf
import os
import csv
from gridworld import gameEnv
from Model import Qnetwork
from helper import *


env = gameEnv(partial=True, size=9)
e = 0.01
num_episodes = 10000
load_model = True
path = "./drqn"
h_size = 512
h_size = 512
max_epLength = 50
time_per_step = 1
summaryLength = 100

tf.reset_default_graph()
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQN = Qnetwork(h_size, cell, 'main')
targetQN = Qnetwork(h_size, cellT, 'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

# Write the first line of the master log-file for the Control Center
with open('./Center/log.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(['Episode', 'Length', 'Reward', 'IMG', 'LOG', 'SAL'])
with tf.Session() as sess:
    if load_model:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(init)

    for i in range(num_episodes):
        episodeBuffer = []
        # Reset environment and get first new observation
        sP = env.reset()
        s = processState(sP)
        d = False
        rAll = 0
        j = 0
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))
        # The Q-Network
        while j < max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e:
                state1 = sess.run(mainQN.rnn_state,
                                  feed_dict={mainQN.scalarInput: [s / 255.0], mainQN.trainLength: 1,
                                             mainQN.state_in: state, mainQN.batch_size: 1})
                a = np.random.randint(0, 4)
            else:
                a, state1 = sess.run([mainQN.predict, mainQN.rnn_state],
                                     feed_dict={mainQN.scalarInput: [s / 255.0], mainQN.trainLength: 1,
                                                mainQN.state_in: state, mainQN.batch_size: 1})
                a = a[0]
            s1P, r, d = env.step(a)
            s1 = processState(s1P)
            total_steps += 1
            episodeBuffer.append(
                np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.
            rAll += r
            s = s1
            sP = s1P
            state = state1
            if d:
                break

        bufferArray = np.array(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        # Periodically save the model.
        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print(total_steps, np.mean(rList[-summaryLength:]), e)
            saveToCenter(i, rList, jList, np.reshape(np.array(episodeBuffer), [len(episodeBuffer), 5]),
                         summaryLength, h_size, sess, mainQN, time_per_step)
print("Percent of successful episodes: " + str(sum(rList) / num_episodes) + "%")
