
# coding: utf-8

# In[1]:

from collections import deque, namedtuple
import itertools
import numpy as np
from matplotlib import pyplot as plt
import gym
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop, SGD
from keras import backend as K
from keras.preprocessing import sequence
from copy import deepcopy


# In[2]:


#set hyperparameters
MINI_BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1000000
AGENT_HISTORY_LENGTH = 1 ## max length
TARGET_NETWORK_UPDATE_FREQUENCY = 5000
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = 100
LEARNING_RATE = 0.00025
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 20000
REPLAY_START_SIZE = 1000
NUM_EPISODES = 20000#####5000


# In[3]:


#setup game env
GAME = "CartPole-v0"
env = gym.envs.make(GAME)
NUMBER_OF_ACTIONS = env.action_space.n
observation = env.reset()


# In[4]:

# define the loss function

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def cliped_mean_squared_error(y_true, y_pred):
    return K.clip(K.mean(K.square(y_pred - y_true), axis=-1), -1, 1)


# In[5]:

# build the model
# input shape is AGENT_HISTORY_LENGT * observation.shape[0]
# output shape is NUMBER_OF_ACTIONS
# using 1 relu hidden layer 
# mean_squared_error as a loss function and 0.0005 for learning rate
def build_model():
    
    model = Sequential()
    input_shape = (AGENT_HISTORY_LENGTH*observation.shape[0])
    model.add(Dense(256,input_dim=input_shape))
    model.add(Activation('relu'))
    model.add(Dense(NUMBER_OF_ACTIONS))
    model.add(Activation('linear'))
    model.compile(loss=mean_squared_error, optimizer=RMSprop(lr=LEARNING_RATE))
    return model

def build_model_LSTM():
    
    model = Sequential()
    input_shape = (1,AGENT_HISTORY_LENGTH,2) ##observation.shape[0])
    model.add(LSTM(128, batch_input_shape=input_shape,  return_sequences=False, stateful=True))
    model.add(Dense(NUMBER_OF_ACTIONS*8))
    model.add(Activation('relu'))
    model.add(Dense(NUMBER_OF_ACTIONS))
    model.add(Activation('linear'))
    model.compile(loss=mean_squared_error, optimizer='adam')
    return model



# In[ ]:




# In[ ]:


# Initialize everything
episode_rewards = np.zeros(NUM_EPISODES)
episode_lengths = np.zeros(NUM_EPISODES)
loss = np.zeros(NUM_EPISODES)
total_frame = 0
all_frame = 0
max_reward = 0
max_ep = 0


print "ver :07"
# The epsilon decay schedule
epsilons = np.linspace(INITIAL_EXPLORATION, FINAL_EXPLORATION, FINAL_EXPLORATION_FRAME)

# build model
train_model = build_model_LSTM()
temp_model = build_model_LSTM()
temp_model.set_weights(train_model.get_weights())


# In[ ]:

for i_episode in xrange(NUM_EPISODES):

    
    # init 
    # state history
    state_history = deque();
    q_value_history = deque();
    ##for _ in xrange(AGENT_HISTORY_LENGTH):
    ##    state_history.append([0,0])
    obs = env.reset()
    train_model.reset_states()
    temp_model.set_weights(train_model.get_weights())
    obs = [obs[i] for i in [0,2]]
    state_history.append(obs)

    count_ran = 0
    count_q = 0 
    
    # predict first one
    temp_model.predict(np.array([[np.array(obs)]]))
    
    for t in itertools.count():
        
        
        # step random action
        action = 0;
        # Calculate q values 
        q_values_train = train_model.predict(np.array([[np.array(obs)]]))  
        
        if np.random.random() < epsilons[min(total_frame,FINAL_EXPLORATION_FRAME-1)]:
                count_ran += 1
                action = np.random.randint(NUMBER_OF_ACTIONS)       
        else:
                count_q += 1
                q_values = q_values_train[0]
                action = np.argmax(q_values)


        print action,
        next_state, reward, done, info = env.step(action)
       

        
        # set the negative reward when the game end  
        if done == True :
            reward = -1

        # clip reward [-5,1]
        reward = max(-5, min(1, reward))

        # Update statistics
        all_frame += 1
        
        total_frame += 1
        episode_rewards[i_episode] += reward
        episode_lengths[i_episode] = t
        
        
        
        # save data for train network    
        
        # Calculate q values and targets
        # predic Q hat - target network from next obs    
        
        # append next state
        if not done :
            obs_next = [next_state[i] for i in [0,2]]
            state_history.append(obs_next)
        obs = obs_next
        
        q_values_target = temp_model.predict(np.array([[np.array(obs_next)]])) 
        
        if done :
            q_values_update = reward
        else :
            q_values_update = reward + DISCOUNT_FACTOR * np.amax(q_values_target)
           

        q_values_train[0][action] = q_values_update
        q_value_history.append(q_values_train)
        
        # check if terminated
        if done or t > 500:
            break
        
    # reset states
    train_model.reset_states()
    
    
    while True:
        try:
            obs = state_history.popleft()
            q_value = q_value_history.popleft()
            # train network    

            # Perform gradient descent update
            
            history =  train_model.fit(np.array([np.array([obs])]), q_value, batch_size=1, nb_epoch=1, verbose=0, shuffle=False)

        except IndexError:
            break
         
       
    # update target network
   ## if i_episode% 10 == 0 :
   ##     target_model.set_weights(train_model.get_weights()) 
    
    if max_reward < episode_rewards[i_episode] :
        max_reward = episode_rewards[i_episode]
        max_ep = i_episode
    # print statistics    
    print 'Done Episode:%i  reward:%i  random_action:%i  predict_action:%i' % (i_episode,episode_rewards[i_episode],count_ran, count_q)

# save model    

# print statistics  
print "max score:{} at episode:{}".format(max_reward,max_ep)



# In[ ]:




# In[ ]:




# In[ ]:



