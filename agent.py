from __future__ import print_function, division
import os
import time
import random
import numpy as np
from base import BaseModel
from replay_memory import ReplayMemory
from utils import save_pkl, load_pkl
import tensorflow as tf
import matplotlib.pyplot as plt



def attention_module(inputs, attention_size):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN hidden states to a fixed size embedding vector.

    :param inputs: The input sequences packed as a dense tensor of shape [batch_size, n_hidden].
                   Note: the hidden size should match the attention size.
    :param attention_size: Linear size of the attention weights.
    :return: Output tensor as a dense tensor with shape [batch_size, attention_size].
    """
    # In order to compute 'v' * 'W', the weights for each sequence, we use tf.map_fn to iterate
    # through sequences and compute for each.
    print("Inputs shape: ", inputs.shape)

    sequence_length = inputs.shape[1].value  # the size of hidden each element of the sequence
    w_omega = tf.Variable(tf.random_normal([attention_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each sequence element independently
        v = tf.tanh(tf.matmul(tf.reshape(inputs, [-1, sequence_length]), w_omega) + tf.reshape(b_omega, [1, -1]))

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.matmul(v, tf.reshape(u_omega, [-1, 1]))
    exps = tf.reshape(tf.exp(vu), [-1, sequence_length])
    alphas = exps / tf.reshape(tf.reduce_sum(exps, 1), [-1, 1])

    # Output of attention mechanism is a weighted sum with weights in alpha
    output = tf.reduce_sum(inputs * tf.reshape(alphas, [-1, sequence_length, 1]), 1)
    print("Sequence length: ", sequence_length)

    return output



# Agent类从BaseModel类继承，表示它有保存和加载模型等基本功能。它的主要职责是使用强化学习算法进行决策，并与环境交互。
# The Agent class inherits from the BaseModel class, indicating that it has basic functions such as saving and loading models.
# Its main responsibility is to use reinforcement learning algorithms to make decisions and to interact with the environment.

class Agent(BaseModel):
    def __init__(self, config, environment, sess):
        self.sess = sess
        self.weight_dir = 'weight'        
        self.env = environment
        #self.history = History(self.config)
        model_dir = './Model/a.model'
        self.memory = ReplayMemory(model_dir) 
        self.max_step = 100000 
        self.RB_number = 20
        self.num_vehicle = len(self.env.vehicles)
        self.action_all_with_power = np.zeros([self.num_vehicle, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.action_all_with_power_training = np.zeros([20, 3, 2],dtype = 'int32')   # this is actions that taken by V2V links with power
        self.reward = []
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 500000
        self.target_q_update_step = 100
        self.discount = 0.5
        self.double_q = True
        self.build_dqn()          
        self.V2V_number = 3 * len(self.env.vehicles)    # every vehicle need to communicate with 3 neighbors  
        self.training = True
        #self.actions_all = np.zeros([len(self.env.vehicles),3], dtype = 'int32')
    def merge_action(self, idx, action):
        self.action_all_with_power[idx[0], idx[1], 0] = action % self.RB_number
        self.action_all_with_power[idx[0], idx[1], 1] = int(np.floor(action/self.RB_number))
    # 这个函数是将动作与对应的资源块（RB）编号合并，形成了一个执行的动作。
    # This function is to merge the action with the corresponding resource block (RB) number to form an executed action.

    def get_state(self, idx):
    # ===============
    #  Get State from the environment
    # =============
        vehicle_number = len(self.env.vehicles)
        V2V_channel = (self.env.V2V_channels_with_fastfading[idx[0],self.env.vehicles[idx[0]].destinations[idx[1]],:] - 80)/60
        V2I_channel = (self.env.V2I_channels_with_fastfading[idx[0], :] - 80)/60
        V2V_interference = (-self.env.V2V_Interference_all[idx[0],idx[1],:] - 60)/60
        NeiSelection = np.zeros(self.RB_number)
        for i in range(3):
            for j in range(3):
                if self.training:
                    NeiSelection[self.action_all_with_power_training[self.env.vehicles[idx[0]].neighbors[i], j, 0 ]] = 1
                else:
                    NeiSelection[self.action_all_with_power[self.env.vehicles[idx[0]].neighbors[i], j, 0 ]] = 1
                   
        for i in range(3):
            if i == idx[1]:
                continue
            if self.training:
                if self.action_all_with_power_training[idx[0],i,0] >= 0:
                    NeiSelection[self.action_all_with_power_training[idx[0],i,0]] = 1
            else:
                if self.action_all_with_power[idx[0],i,0] >= 0:
                    NeiSelection[self.action_all_with_power[idx[0],i,0]] = 1
        time_remaining = np.asarray([self.env.demand[idx[0],idx[1]] / self.env.demand_amount])
        load_remaining = np.asarray([self.env.individual_time_limit[idx[0],idx[1]] / self.env.V2V_limit])
        #print('shapes', time_remaining.shape,load_remaining.shape)
        return np.concatenate((V2I_channel, V2V_interference, V2V_channel, NeiSelection, time_remaining, load_remaining))#,time_remaining))
        #return np.concatenate((V2I_channel, V2V_interference, V2V_channel, time_remaining, load_remaining))#,time_remaining))
    # 从环境中获取状态，包括车辆数量、V2V通道、V2I通道、V2V干扰、邻居选择、剩余时间和负载等。
    # Status is obtained from the environment, including the number of vehicles, V2V channels, V2I channels, V2V interference, neighbor selection, residual time, and load, etc.

    def predict(self, s_t,  step, test_ep = False):
        # ==========================
        #  Select actions
        # ======================
        ep = 1/(step/1000000 + 1)
        if random.random() < ep and test_ep == False:   # epsion to balance the exporation and exploition
            action = np.random.randint(60)
        else:          
            action =  self.q_action.eval({self.s_t:[s_t]})[0] 
        return action
# 这个函数通过在当前状态下选择一个动作。如果随机选择的数小于epsilon（一个用于控制贪婪程度的参数），则随机选择一个动作；否则，根据当前状态选择最佳动作。
# This function works by selecting an action in the current state.
# If the randomly selected number is less than epsilon (a parameter controlling the degree of greed), randomly select one action;
# otherwise, select the best action according on the current state.
    def observe(self, prestate, state, reward, action):
        # -----------
        # Collect Data for Training 
        # ---------
        self.memory.add(prestate, state, reward, action) # add the state and the action and the reward to the memory
        #print(self.step)
        if self.step > 0:
            if self.step % 100 == 0:
                #print('Training')
                self.q_learning_mini_batch()            # training a mini batch
                #self.save_weight_to_pkl()
            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                #print("Update Target Q network:")
                self.update_target_q_network()           # ?? what is the meaning ??
#这个函数用于观察从执行一个动作后得到的结果，包括新的状态和奖励，并将其加入到回放内存中。
# 然后，每50步执行一次训练，并每100步更新目标Q网络。
# This function is used to observe the results obtained from after performing an action, including the new states and rewards, and to add it to the playback memory.
# Then, training was performed every 50 steps and the target Q network was updated every 100 steps.

    def train(self):        
        num_game, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        max_avg_ep_reward = 0
        ep_reward, actions = [], []        
        mean_big = 0
        number_big = 0
        mean_not_big = 0
        number_not_big = 0
        self.env.new_random_game(20)
        for self.step in (range(0, 40000)): # need more configuration
            if self.step == 0:                   # initialize set some varibles
                num_game, self.update_count,ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_reward, actions = [], []               
                
            # prediction
            # action = self.predict(self.history.get())
            if (self.step % 2000 == 1):
                self.env.new_random_game(20)
            print(self.step)
            state_old = self.get_state([0,0])
            #print("state", state_old)
            self.training = True
            for k in range(1):
                for i in range(len(self.env.vehicles)):              
                    for j in range(3): 
                        state_old = self.get_state([i,j]) 
                        action = self.predict(state_old, self.step)                    
                        #self.merge_action([i,j], action)   
                        self.action_all_with_power_training[i, j, 0] = action % self.RB_number
                        self.action_all_with_power_training[i, j, 1] = int(np.floor(action/self.RB_number))                                                    
                        reward_train = self.env.act_for_training(self.action_all_with_power_training, [i,j]) 
                        state_new = self.get_state([i,j]) 
                        self.observe(state_old, state_new, reward_train, action)
            if (self.step % 100 == 0) and (self.step > 0):
                # testing 
                self.training = False
                number_of_game = 10
                if (self.step % 10000 == 0) and (self.step > 0):
                    number_of_game = 50 
                if (self.step == 38000):
                    number_of_game = 100               
                V2I_Rate_list = np.zeros(number_of_game)
                Fail_percent_list = np.zeros(number_of_game)
                for game_idx in range(number_of_game):
                    self.env.new_random_game(self.num_vehicle)
                    test_sample = 200
                    Rate_list = []
                    print('test game idx:', game_idx)
                    for k in range(test_sample):
                        action_temp = self.action_all_with_power.copy()
                        for i in range(len(self.env.vehicles)):
                            self.action_all_with_power[i,:,0] = -1
                            sorted_idx = np.argsort(self.env.individual_time_limit[i,:])          
                            for j in sorted_idx:                   
                                state_old = self.get_state([i,j])
                                action = self.predict(state_old, self.step, True)
                                self.merge_action([i,j], action)
                            if i % (len(self.env.vehicles)/10) == 1:
                                action_temp = self.action_all_with_power.copy()
                                reward, percent = self.env.act_asyn(action_temp) #self.action_all)            
                                Rate_list.append(np.sum(reward))
                        #print("actions", self.action_all_with_power)
                    V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
                    Fail_percent_list[game_idx] = percent
                    #print("action is", self.action_all_with_power)
                    print('failure probability is, ', percent)
                    #print('action is that', action_temp[0,:])
                self.save_weight_to_pkl()
                print ('The number of vehicle is ', len(self.env.vehicles))
                print ('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
                print('Mean of Fail percent is that ', np.mean(Fail_percent_list))                   
                #print('Test Reward is ', np.mean(test_result))
# 这个函数是主训练循环，根据预定义的步骤进行训练。对每一个步骤，智能体都会选择和执行一个动作，并观察结果。
# 每2000步，会进行一次新的游戏，每5000步或者在特定的步骤，进行一次测试。
# This function is the main training loop, trained according to the predefined steps.
# For each step, the agent selects and performs an action, and observes the outcome.
# Every 2,000 steps, a new game, every 5,000 steps or at specific steps.



    def q_learning_mini_batch(self):

        # Training the DQN model
        # ------ 
        #s_t, action,reward, s_t_plus_1, terminal = self.memory.sample() 
        s_t, s_t_plus_1, action, reward = self.memory.sample()  

        print(s_t_plus_1.shape)
        print('对的')
        #print() 
        #print('samples:', s_t[0:10], s_t_plus_1[0:10], action[0:10], reward[0:10])        
        t = time.time()        
        if self.double_q:       #double Q learning  
            test1 =  {self.s_t: s_t_plus_1}
            pred_action = self.q_action.eval({self.s_t: s_t_plus_1})       
            q_t_plus_1_with_pred_action = self.target_q_with_idx.eval({self.target_s_t: s_t_plus_1, self.target_q_idx: [[idx, pred_a] for idx, pred_a in enumerate(pred_action)]})            
            target_q_t =  self.discount * q_t_plus_1_with_pred_action + reward
        else:
            q_t_plus_1 = self.target_q.eval({self.target_s_t: s_t_plus_1})         
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = self.discount * max_q_t_plus_1 +reward
        _, q_t, loss,w = self.sess.run([self.optim, self.q, self.loss, self.w], {self.target_q_t: target_q_t, self.action:action, self.s_t:s_t, self.learning_rate_step: self.step}) # training the network
        
        print('loss is ', loss)
        self.total_loss += loss
        self.total_q += q_t.mean()
        self.update_count += 1
# 函数是用来执行基于经验回放的Q学习的训练步骤的。
# 在double Q学习中，我们使用了两个网络，一个用来选择动作，另一个用来估计那个动作的Q值。
# 计算目标Q值后，我们用这些目标Q值来训练DQN网络。

    def build_dqn(self): 
    # --- Building the DQN -------
        self.w = {}
        self.t_w = {}        
        
        initializer = tf. truncated_normal_initializer(0, 0.02)
        activation_fn = tf.nn.relu
        # Parameters
        n_hidden_1 = 500
        n_hidden_2 = 250
        n_hidden_3 = 120
        n_input = 82
        n_output = 60
        n_attention = 82

        def encoder(x):
            attention_output = attention_module(x, n_attention)
            # attention_output = attention_module(x, n_attention)
            print("Shape of attention_output:", attention_output.shape) 
            weights = {
                'encoder_h1': tf.Variable(tf.truncated_normal([n_attention, n_hidden_1], stddev=0.1)),
                'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2], stddev=0.1)),
                'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3], stddev=0.1)),
                'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_output], stddev=0.1)),
                'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1], stddev=0.1)),
                'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2], stddev=0.1)),
                'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3], stddev=0.1)),
                'encoder_b4': tf.Variable(tf.truncated_normal([n_output], stddev=0.1)),
            }
            layer_1 = tf.nn.relu(tf.add(tf.matmul(attention_output, weights['encoder_h1']), weights['encoder_b1']))
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['encoder_h2']), weights['encoder_b2']))
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, weights['encoder_h3']), weights['encoder_b3']))
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))
            # layer_4 = tf.nn.softmax(tf.add(tf.matmul(layer_3, weights['encoder_h4']), weights['encoder_b4']))
            return tf.expand_dims(layer_4, 1), weights
            # return layer_4, weights


        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder('float32',[None, n_input])            
            self.q, self.w = encoder(self.s_t)
            print("Shape of self.q:", self.q.shape)
            print(self.q)
            # self.q = tf.squeeze(self.q, axis=1)# Remove the extra dimension
            self.q_action = tf.argmax(self.q, dimension = 1)
        with tf.variable_scope('target'):
            self.target_s_t = tf.placeholder('float32', [None, n_input])
            self.target_q, self.target_w = encoder(self.target_s_t)
            self.target_q_idx = tf.placeholder('int32', [None,None], 'output_idx')
            self.target_q_with_idx = tf.gather_nd(self.target_q, self.target_q_idx)
        with tf.variable_scope('pred_to_target'):
            self.t_w_input = {}
            self.t_w_assign_op = {}
            for name in self.w.keys():
                print('name in self w keys', name)
                self.t_w_input[name] = tf.placeholder('float32', self.target_w[name].get_shape().as_list(),name = name)
                self.t_w_assign_op[name] = self.target_w[name].assign(self.t_w_input[name])       
        
        def clipped_error(x):
            try:
                return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
            except:
                return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

        with tf.variable_scope('optimizer'):
            self.target_q_t = tf.placeholder('float32', None, name='target_q_t')
            self.action = tf.placeholder('int32',None, name = 'action')
            action_one_hot = tf.one_hot(self.action, n_output, 1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q * action_one_hot, reduction_indices = 1, name='q_acted')
            self.delta = self.target_q_t - q_acted
            self.global_step = tf.Variable(0, trainable=False)
            self.loss = tf.reduce_mean(tf.square(self.delta), name = 'loss')
            self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')
            self.learning_rate_op = tf.maximum(self.learning_rate_minimum, tf.train.exponential_decay(self.learning_rate, self.learning_rate_step, self.learning_rate_decay_step, self.learning_rate_decay, staircase=True))
            self.optim = tf.train.RMSPropOptimizer(self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss) 
        
        tf.initialize_all_variables().run()
        self.update_target_q_network()
# 函数是用来建立DQN网络的。
# 这个函数中定义了两个网络，一个是用于预测的网络（也就是我们通常所说的DQN网络），另一个是目标网络。


    def update_target_q_network(self):    
        for name in self.w.keys():
            self.t_w_assign_op[name].eval({self.t_w_input[name]: self.w[name].eval()})       
# 函数是用来更新目标网络参数的。在DQN中，我们通常会定期将预测网络的参数复制到目标网络，保持学习的稳定性。

    def save_weight_to_pkl(self): 
        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        for name in self.w.keys():
            save_pkl(self.w[name].eval(), os.path.join(self.weight_dir,"%s.pkl" % name))       
    def load_weight_from_pkl(self):
        with tf.variable_scope('load_pred_from_pkl'):
            self.w_input = {}
            self.w_assign_op = {}
            for name in self.w.keys():
                self.w_input[name] = tf.placeholder('float32')
                self.w_assign_op[name] = self.w[name].assign(self.w_input[name])
        for name in self.w.keys():
            self.w_assign_op[name].eval({self.w_input[name]:load_pkl(os.path.join(self.weight_dir, "%s.pkl" % name))})
        self.update_target_q_network()   
      
    def play(self, n_step = 100, n_episode = 100, test_ep = None, render = False):
        number_of_game = 100
        V2I_Rate_list = np.zeros(number_of_game)
        Fail_percent_list = np.zeros(number_of_game)
        self.load_weight_from_pkl()
        self.training = False


        for game_idx in range(number_of_game):
            self.env.new_random_game(self.num_vehicle)
            test_sample = 200
            Rate_list = []
            print('test game idx:', game_idx)
            print('The number of vehicle is ', len(self.env.vehicles))
            time_left_list = []
            power_select_list_0 = []
            power_select_list_1 = []
            power_select_list_2 = []

            for k in range(test_sample):
                action_temp = self.action_all_with_power.copy()
                for i in range(len(self.env.vehicles)):
                    self.action_all_with_power[i, :, 0] = -1
                    sorted_idx = np.argsort(self.env.individual_time_limit[i, :])
                    for j in sorted_idx:
                        state_old = self.get_state([i, j])
                        time_left_list.append(state_old[-1])
                        action = self.predict(state_old, 0, True)
                        '''
                        if state_old[-1] <=0:
                            continue
                        power_selection = int(np.floor(action/self.RB_number))
                        if power_selection == 0:
                            power_select_list_0.append(state_old[-1])

                        if power_selection == 1:
                            power_select_list_1.append(state_old[-1])
                        if power_selection == 2:
                            power_select_list_2.append(state_old[-1])
                        '''
                        self.merge_action([i, j], action)
                    if i % (len(self.env.vehicles) / 10) == 1:
                        action_temp = self.action_all_with_power.copy()
                        reward, percent = self.env.act_asyn(action_temp)  # self.action_all)
                        Rate_list.append(np.sum(reward))
                # print("actions", self.action_all_with_power)
            '''
            number_0, bin_edges = np.histogram(power_select_list_0, bins = 10)

            number_1, bin_edges = np.histogram(power_select_list_1, bins = 10)

            number_2, bin_edges = np.histogram(power_select_list_2, bins = 10)


            p_0 = number_0 / (number_0 + number_1 + number_2)
            p_1 = number_1 / (number_0 + number_1 + number_2)
            p_2 = number_2 / (number_0 + number_1 + number_2)

            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_0, 'b*-', label='Power Level 23 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_1, 'rs-', label='Power Level 10 dB')
            plt.plot(bin_edges[:-1]*0.1 + 0.01, p_2, 'go-', label='Power Level 5 dB')
            plt.xlim([0,0.12])
            plt.xlabel("Time left for V2V transmission (s)")
            plt.ylabel("Probability of power selection")
            plt.legend()
            plt.grid()
            plt.show()
            '''
            V2I_Rate_list[game_idx] = np.mean(np.asarray(Rate_list))
            Fail_percent_list[game_idx] = percent

            print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list[0:game_idx] ))
            print('Mean of Fail percent is that ',percent, np.mean(Fail_percent_list[0:game_idx]))
            # print('action is that', action_temp[0,:])

        print('The number of vehicle is ', len(self.env.vehicles))
        print('Mean of the V2I rate is that ', np.mean(V2I_Rate_list))
        print('Mean of Fail percent is that ', np.mean(Fail_percent_list))
        # print('Test Reward is ', np.mean(test_result))
	




