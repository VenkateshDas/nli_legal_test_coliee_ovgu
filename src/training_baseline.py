# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 21 18:34:01 2019

@author: Sachin Nandakumar
"""

'''#######################################################
                        TRAINING
#######################################################'''

import os
import h5py
import datetime
import numpy as np
from data_parser import data_parser_for_baseline as dp

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib import rnn
from keras.utils.np_utils import to_categorical
#import tensorflow.python.util.deprecation as deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

'''############################################################
Location of file(s):
    1. required to run the program
    2. required to save the models & states to
############################################################'''

PREPROCESSED_TRAIN_SET = "../data/preprocessed_data/preprocessed_training_set.json"
SAVE_MODEL_TO = "../models/baseline/"
SAVE_STATES_TO = "../states/baseline/states"
SAVE_LOGS_TO = "../tensorBoardLogs/baseline/"
TRAINING_LOG = "../logs/baseline/training_performance_log.txt"

'''############################################################
Get data (premise, hypothesis, labels) for training
############################################################'''

premise_hypo_pair, correct_labels = dp.get_data(PREPROCESSED_TRAIN_SET)

X_train = premise_hypo_pair
y_train = to_categorical(correct_labels)

# Shuffle the dataset with different random_state to perform stratified split of training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1, stratify=y_train)

'''############################################################
Define & initialize constants for lstm architecture
    > constants for network architecture
    > for network optimization
############################################################'''

# Training Parameters
learning_rate = 1e-06
#learning_rate = 0.0001
num_input = X_train.shape[2]            # dimension of each sentence 
timesteps = X_train.shape[1]            # timesteps
num_hidden = {1: 100, 2: 64}            # dictionary that defines number of neurons per layer 
num_classes = 2                         # total number of classes
num_layers = 1                          # desired number of LSTM layers
    
weight_decay = 1e-06                    # hyperparameter for regularizer

del premise_hypo_pair, correct_labels   # delete unused variables to free RAM


'''#######################################################
> Reset tensorflow graphs
> Define network input placeholders
> Define initializer and weights
#######################################################'''

# Clears the default graph stack and resets the global default graph. The default graph is a property of the current thread.
# Once a graph is created, all placeholders, variables and any elements are actually part of the current thread.
# If we need to re-execute any of the tensorflow related code again, you need to reset the graph to its default state.
tf.compat.v1.reset_default_graph() 

# Declare placeholders for input and labels that is required for tensor graph
X = tf.compat.v1.placeholder("float", [None, timesteps, num_input])
y = tf.compat.v1.placeholder("float", [None, num_classes])

initializer = tf.compat.v1.keras.initializers.VarianceScaling() # He initialization. Better for ReLu (Based on Delving Deep into Rectifiers: https://arxiv.org/pdf/1502.01852v1.pdf)
# initializer = tf.contrib.layers.xavier_initializer()

fc_weights = {
        'out' : tf.Variable(initializer(([2*num_hidden[1], num_classes])), name='w_out')    # output weights for applying softmax
        }

fc_biases = {
        'out' : tf.Variable(tf.zeros([num_classes]), name='b_out')                          # output bias
        }


'''#######################################################
Define BiLSTM network architecture
#######################################################'''
   
def BiRNN(x, weights, bias):
    '''
        BiRNN: Defines the architecture of LSTM network for training
        Args: 
                x:          premise_hypothesis pair
                weights:    weights required to apply relu activation function over hidden layer and softmax activation over output layer 
                bias:       bias corresponding to the weights.
        
        Returns:
            1. muladd() applied over last outputs with corresponding weights and bias
            2. concatenated forward and backward cell states
            3. whole rnn output
    '''
    x = tf.unstack(x, timesteps, 1)
    output = x   
    
    for i in range(num_layers):
       lstm_fw_cell = rnn.BasicLSTMCell(num_hidden[i+1], forget_bias=1.0, activation=tf.nn.relu)           # define forward lstm cell with hidden cells
       lstm_fw_cell = rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=0.5)                               # define dropout over hidden forward lstm cell
       lstm_bw_cell = rnn.BasicLSTMCell(num_hidden[i+1], forget_bias=1.0, activation=tf.nn.relu)           # define backward lstm cell with hidden cells
       lstm_bw_cell = rnn.DropoutWrapper(lstm_bw_cell,  output_keep_prob=0.5)                              # define dropout over hidden backward lstm cell
            
       with tf.compat.v1.variable_scope('lstm'+str(i)):
            try:
                output, state_fw, state_bw = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, output, dtype=tf.float32)
            except Exception: # Old TensorFlow version only returns outputs not states
                output = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, output, dtype=tf.float32)
            
            
            # Venky: rnn cell output  --> currently this is not used for LSTMVis
            outputs = tf.unstack(output, timesteps, 0)
            outputs = tf.transpose(outputs, perm=[1, 0, 2]) 
    
    return tf.add(tf.matmul(output[-1], weights['out']), bias['out']), outputs
    
'''############################################################
Define: activation, loss, regularization, optimizer,
        prediction, accuracy, gradient clipping
############################################################'''

with tf.name_scope("output"):
    logits, output = BiRNN(X, fc_weights, fc_biases)
    prediction = tf.nn.softmax(logits, name='prediction')   # applies softmax over BiRNN output to calculate predicted values
#tf.compat.v1.summary.histogram("prediction", prediction)    # write predicted values to tensorboard summary (histogram visualization)


with tf.name_scope("loss"):
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))      # calculate loss 
    tf.compat.v1.summary.scalar('loss_op', loss_op)                                                 # write loss values to tensorboard summary (histogram visualization)
    regularizer= tf.nn.l2_loss(fc_weights['out'])                                                   # apply regularizer over output weights
    loss_op = loss_op + weight_decay * regularizer                                                  # add regularization term with loss.
    
#    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)                                     # apply Adam Optimizer for loss optimization
    gvs = optimizer.compute_gradients(loss_op)                                                      # fetch gradient values
    capped_gvs = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in gvs]                    # clip each gradient value within the limit
    
    train_op = optimizer.apply_gradients(gvs)                                                       # applied clipped gradients
    #     train_op = optimizer.minimize(loss_op)
   
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))                       # obtain correct predictions on comparison with actual labels
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name="accuracy")               # mean of correct predictions
tf.compat.v1.summary.scalar('accuracy', accuracy)


'''#######################################################
Begin training
#######################################################'''

def save_LSTM_states(states_inter, state_val, SAVE_STATES_TO):
    final_states = []
    states_inter = np.vstack(states_inter)
    final_states.append(states_inter)                       # append training_states to final_states 
    final_states.append(np.array(state_val))                # append validation_states to final_states
    val_1 = final_states[0][0]
    for k in range(len(final_states)):
        for i in range(0,len(final_states[k])):
            temp = final_states[k][i]
            val_1 = np.concatenate((val_1,temp),axis=0)
    print('\nSaving LSTM states...')
    with h5py.File(SAVE_STATES_TO, 'w') as hf:
        hf.create_dataset("d1",  data= val_1)
    print('LSTM states saved to {}'.format(SAVE_STATES_TO))

def run_train(session, train_x, train_y):
    '''
    Description:    Trains the BiLSTM model with given training set in batches and returns final training results and states 
    Input:          1. session: Tensorflow session
                    2. train_x: training set of padded premise_hypothesis sequences
                    3. train_y: Two column binary labels that corresponds to the train_x
    Output:         List of training accuracy and loss results, List of final training and validation states
    '''
    print("\nStart training")
    ###################################################
    # initialization of local variables and lists:
    acc_results = []
    loss_results = []
    train_counter = 0
    validation_counter = 0
    
    training_steps = 10000  # epochs
    batch_size = 128        # batch size
    display_step = 10       # displays 
    
    #for early stopping :    
    best_loss_val=1000000   # initializing best validation loss to a higher value.
    best_train_acc = 0      # best training accuracy
    last_improvement=0      # a counter which keeps the record of since when (timesteps/iterations) last improvement was seen
    patience= 10            # the number of epochs without improvement you allow before training should be aborted
    average_loss_baseline = 0.74
    # since the values are updated every 10th iteration, the stopping limit becomes: (patience * 10)
    
    costs = []              # validation costs history
    costs_inter=[]          # intermediate validation costs. These values are only used as a log to keep track of the costs.
    best_loss_observed_epoch = 0
    
    ###################################################
    
    session.run(tf.compat.v1.global_variables_initializer())                        # initialize all variables using session
    for epoch in range(1, training_steps + 1):                                      # training iterations
        train_x, train_y = shuffle(train_x, train_y)
        inner_split = train_x.shape[0] // batch_size                                # creating batches 
        states_inter = []
        
        for i in range(inner_split + 1):
            batch_x = train_x[i*batch_size:(i+1)*batch_size]                        # generating batches of X_train
            batch_y = train_y[i*batch_size:(i+1)*batch_size]                        # generating batches of y_train
            session.run(train_op, feed_dict={X: batch_x, y: batch_y})
            
            if epoch == 1 or epoch % display_step == 0:                             # print and save necessary information about training only at an interval of 'display_step' number of steps to reduce computational complexity
                
                state_train = session.run([output], feed_dict={X: batch_x, y: batch_y})     # extract states for each batch-wise training inputs
                states_inter.append(np.array(state_train)[0])
            
                if i == inner_split:                                                # last batch split of the selected epoch 
                    summary, loss_train, acc_train = session.run([merged, loss_op, accuracy], feed_dict={X: batch_x, y: batch_y})
                    train_writer.add_summary(summary, train_counter)
                    
                    summary, loss_val, acc_val, state_val = session.run([merged, loss_op, accuracy, output], feed_dict={X: X_val, y: y_val})
                    validation_writer.add_summary(summary, validation_counter)
                    train_counter+=display_step
                    validation_counter+=display_step
                    
                    print("Epoch {}, Batch Split {}".format(epoch, i+1) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss_train) + ", Minibatch Training Accuracy= " + \
                      "{:.3f}".format(acc_train))
                    print(" Validation Loss = {:.4f}".format(loss_val) + ", Validation Accuracy= {:.3f}".format(acc_val))
                    
                    acc_results.append(acc_train)
                    loss_results.append(loss_train)
                    
                    #...... BEGIN EARLY STOPPING EVALUATION ......
                    
                    # CONDITION: 
                    # 1. If validation loss has not decreased since 'patience' steps
                    #   1.1. If the average of last 'patience' iterations are less than 'average_loss_baseline'
                    
                    costs_inter.append(loss_val)            # append validation loss to costs_inter
                    
                    if loss_val < best_loss_val:            # if improved validation loss found
                        best_loss_val = loss_val            # set current validation loss to best_loss_val
                        best_train_acc = acc_train          # set current training accuracy to best_train_acc
                        best_val_acc = acc_val              # set current validation accuracy to acc_val
                        costs +=costs_inter                 # append intermediate cost history to costs
                        last_improvement = 0                # reset last_improvement
                        costs_inter= []                     # reset costs_inter
                        best_loss_observed_epoch = epoch
                    else:
                        last_improvement +=1                # else, increment last_improvement
                        
                    if last_improvement > patience:         # if no improvement seen over 'patience' number of steps
                        print("\nNo improvement found during the last {} iterations".format(patience))
                        print('Avg validation loss over this period: ', sum(costs_inter)/len(costs_inter))  
                        
                        if (sum(costs_inter)/len(costs_inter)) > average_loss_baseline:      # if average of validation loss greater than 'average_loss_baseline' (a hyper-parameter to optimize)      
                            # final stopping condition
                            print('Avg validation loss > {}\nStopping optimization'.format(average_loss_baseline))
                            print('Recording training and validation states at cost of early-stopping')
                            save_LSTM_states(states_inter, state_val, SAVE_STATES_TO+'-final.hdf5')
                            return acc_results, loss_results
                        else:                                                   # else, save checkpoint and reset costs_inter and last_improvement
                            print('\nSaving Checkpoint! Avg validation loss < {}'.format(average_loss_baseline))
                            _ = saver.save(session, SAVE_MODEL_TO+"m_{}_{}.ckpt".format(acc_train, acc_val), global_step=epoch)
                            print('<<<Model Checkpoint saved>>>')
                            print('<<<State Checkpoint saved>>>')
                            save_LSTM_states(states_inter, state_val, SAVE_STATES_TO+'-'+str(epoch)+'.hdf5')
                            
                            print('Last improvement was: Training acc = {}, Validation acc = {} observed at {}'.format(best_train_acc, best_val_acc, best_loss_observed_epoch)) # the best result seen before 'no improvements'
                            
                            # to_log = 'Best result: m_{}_{}.ckpt-{}'.format(best_train_acc, best_val_acc, best_loss_observed_epoch)
                            # file_op = open(TRAINING_LOG,"a+") 
                            # file_op.write(to_log + '\n')
                            # file_op.close()
                            
                            print('Continuing Training...\n')
                            costs_inter = []
                            last_improvement = 0
                            best_loss_val = 1000000
                            best_train_acc = 0  
                            
                    
                    #...... END EARLY STOPPING EVALUATION ......
                    
                    if epoch == training_steps:                                 # do not change this intendation to make sure this line run only once and not for each split of the epoch!
                        _ = saver.save(session, SAVE_MODEL_TO+"m_{}_{}.ckpt".format(acc_train, acc_val), global_step=epoch)                         # save model to local
                        print('Recording final training and validation states')
                        # append states to list before ending training
                        save_LSTM_states(states_inter, state_val, SAVE_STATES_TO+'-final.hdf5')
                        print('\nBest result: Training acc = {}, Validation acc = {} observed at {}'.format(best_train_acc, best_val_acc, best_loss_observed_epoch)) # the best result seen before 'no improvements'
                        
    return acc_results, loss_results
        

saver = tf.compat.v1.train.Saver(max_to_keep=10)
with tf.compat.v1.Session() as sess:
    # Log for tensorboard visualization
    logdir = os.path.join(SAVE_LOGS_TO, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(logdir + '/train', sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(logdir + '/validation')
    
    start_time = datetime.datetime.now()
    print('-'*50)
    print('\nSession started at: {}'.format(start_time))
    acc_results, loss_results = run_train(sess, X_train, y_train)
#    summary, loss_val, acc_test, pred_test = sess.run([merged, loss_op, accuracy, prediction, output], feed_dict={X: X_test, y: y_test})
    print('Training performance: Accuracy {}, Loss {}'.format(acc_results[-1], loss_results[-1]))
    end_time = datetime.datetime.now()
    print('Total Execution time: {} minutes'.format(end_time - start_time))