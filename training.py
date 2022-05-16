import numpy as np
import tensorflow as tf
import os
import csv
from gridworld import gameEnv
from Model import Qnetwork, experience_buffer
from helper import *


env = gameEnv(partial=True, size=9)
batch_size = 4
trace_length = 8
update_freq = 5
y = .99
startE = 1
endE = 0.1
anneling_steps = 10000
num_episodes = 10000
pre_train_steps = 10000
load_model = False
path = "./drqn"
h_size = 512
max_epLength = 50
time_per_step = 1
summaryLength = 100
tau = 0.001

tf.reset_default_graph()
# We define the cells for the primary and target q-networks
cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
cellT = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)
mainQN = Qnetwork(h_size, cell, 'main')
targetQN = Qnetwork(h_size, cellT, 'target')

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=5)

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE) / anneling_steps

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
    sess.run(init)

    updateTarget(targetOps, sess)  # Set the target network to be equal to the primary network.
    for i in range(num_episodes):
        episodeBuffer = []
        # Reset environment and get first new observation
        sP = env.reset()
        s = processState(sP)
        d = False
        rAll = 0
        j = 0
        state = (np.zeros([1, h_size]), np.zeros([1, h_size]))  # Reset the recurrent layer's hidden state
        # The Q-Network
        while j < max_epLength:
            j += 1
            # Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
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
            episodeBuffer.append(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    updateTarget(targetOps, sess)
                    # Reset the recurrent layer's hidden state
                    state_train = (np.zeros([batch_size, h_size]), np.zeros([batch_size, h_size]))

                    trainBatch = myBuffer.sample(batch_size, trace_length)  # Get a random batch of experiences.
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={
                        mainQN.scalarInput: np.vstack(trainBatch[:, 3] / 255.0),
                        mainQN.trainLength: trace_length, mainQN.state_in: state_train, mainQN.batch_size: batch_size})
                    Q2 = sess.run(targetQN.Qout, feed_dict={
                        targetQN.scalarInput: np.vstack(trainBatch[:, 3] / 255.0),
                        targetQN.trainLength: trace_length, targetQN.state_in: state_train,
                        targetQN.batch_size: batch_size})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size * trace_length), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    sess.run(mainQN.updateModel,
                             feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0] / 255.0),
                                        mainQN.targetQ: targetQ,
                                        mainQN.actions: trainBatch[:, 1], mainQN.trainLength: trace_length,
                                        mainQN.state_in: state_train, mainQN.batch_size: batch_size})
            rAll += r
            s = s1
            sP = s1P
            state = state1
            if d:
                break

        # Add the episode to the experience buffer
        bufferArray = np.array(episodeBuffer)
        episodeBuffer = list(zip(bufferArray))
        myBuffer.add(episodeBuffer)
        jList.append(j)
        rList.append(rAll)

        # Periodically save the model.
        if i % 1000 == 0 and i != 0:
            saver.save(sess, path + '/model-' + str(i) + '.cptk')
            print("Saved Model")
        if len(rList) % summaryLength == 0 and len(rList) != 0:
            print(total_steps, np.mean(rList[-summaryLength:]), e)
            saveToCenter(i, rList, jList, np.reshape(np.array(episodeBuffer), [len(episodeBuffer), 5]),
                         summaryLength, h_size, sess, mainQN, time_per_step)
    saver.save(sess, path + '/model-' + str(i) + '.cptk')
