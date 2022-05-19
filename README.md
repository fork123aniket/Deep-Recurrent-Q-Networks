# Deep Recurrent Q Networks
This repository demonstrates the implementation of ***Deep Recurrent Q Networks (DRQN)*** for ***Partially Observable*** environments. Utilizing recurrent blocks with ***Deep Q Networks*** can actually make the agent receiving single frames of the environment and the network will be able to change its output depending on the temporal pattern of observations it receives. ***DRQN*** does this by maintaining a hidden state that it computes at every time-step. Furthermore, A brief explaination of ***DRQNs*** for ***partial observability*** can be found [here](https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-6-partial-observability-and-deep-recurrent-q-68463e9aeefc#.gi4xdq8pk).
## Requirements
- `tensorflow`
- `numpy`
## Training Environment Used
- `The Spatially Limited Gridworld Environment`
In this new version of the GridWorld, the agent can only see a single block around it in any direction, whereas the environment itself contains *9x9* blocks. Each episode is fixed at *50* steps, there are four green and two red squares, and when the agent moves to a red or green square, a new one is randomly placed in the environment to replace it.
## Usage
- To train a new network : run `training.py`
- To test a preTrained network : run `test.py`
- To see the DRQNN implementation, please check `Model.py`
- Other imperative helper utilities to properly train the ***Deep Recurrent Q Networks*** can be found in `helper.py` file.
- All hyperparamters to control training and testing of ***DRQNs*** are provided in their respective `.py` files.
## Results
| Gridworld Environment        | Gridworld Results           |
| ------------------------------ |:-----------------------------:|
| ![alt text](https://github.com/fork123aniket/Deep-Recurrent-Q-Networks/blob/main/Images/GridWorld.gif) | ![alt text](https://github.com/fork123aniket/Deep-Recurrent-Q-Networks/blob/main/Images/DRQN%20Reward%20Plot.png) |
