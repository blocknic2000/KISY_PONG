import numpy as np
from collections import deque
import random

import tensorflow as tf
from tensorflow import keras

class my_agent:
    def __init__(self,inp_shape, output_shape,loadmodel=False,trainme=True,filename="pong.keras"):

        self.GAMMA = 0.95  #parameter in deep-Q formula for importance of the future steps
        self.EPSILON = 1.0 #epsilon greedy strategy - startet mit 100% Exploration
        self.EPSILON_MIN = 0.01  # mindestens 1% Exploration bleiben
        self.EPSILON_DECAY = 0.9995  # langsamere Reduktion für besseres Lernen
        self.BATCH_SIZE = 32 #number of random sampled items from memory that are used for a training step
        self.MEMORY_SIZE = 50000  # größerer Memory Buffer für bessere Trainingsdaten
        self.LEARNING_RATE = 0.0001 #kleinere Lernrate für stabileres Training
        self.TRAIN_START = 500  # warte bis 500 Samples gesammelt sind
        self.inp_shape=inp_shape #number of input neurons
        self.output_shape=output_shape #number of output neurons (=number of actions)
        self.step=1
        self.n_update_target_model=1000 # update target model seltener für Stabilität
        self.model_filename=filename
        self.n_save_model=100  # speichere seltener um Performance zu verbessern 

        if not trainme:
            self.EPSILON=0
            self.EPSILON_MIN=0

        if loadmodel:
            self.model = tf.keras.models.load_model(self.model_filename)
        else: 
            self.model = self.build_model(inp_shape, output_shape)

        #Using a target model stabilitzes the training process against overoscillations or slow learning
        # Neuronales Netze 
        self.target_model = self.build_model(inp_shape, output_shape)
        self.target_model.set_weights(self.model.get_weights()) #target model starts with the same weights as trained model

        
        self.memory = deque(maxlen=self.MEMORY_SIZE) #deque automatically limits the length of the memory
        self.loss_fn = keras.losses.Huber() #loss function that is faster than mean squared error
        self.optimizer = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE)  

    # --- Build Q-network ---
    def build_model(self,inp_shape, output_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(inp_shape,)))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dense(output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE))
        return model

    def train(self):
        if len(self.memory) > self.TRAIN_START: 
            if self.step % 100 == 0:
                print(f"\n=== Training Step {self.step} | Memory: {len(self.memory)} | Epsilon: {self.EPSILON:.4f} ===")
            # sample random choice of states and rewards from memory and convert them to numpy arrays
            minibatch = random.sample(self.memory, self.BATCH_SIZE)

            # unpack minibatch and ensure proper dtypes / shapes
            states = np.vstack([data[0] for data in minibatch]).astype(np.float32)
            actions = np.array([data[1] for data in minibatch], dtype=np.int32)
            rewards = np.array([data[2] for data in minibatch], dtype=np.float32)
            next_states = np.vstack([data[3] for data in minibatch]).astype(np.float32)
            done = np.array([data[4] for data in minibatch], dtype=np.float32)

            # use target model to calculate future Q-values / rewards
            next_Q_values = self.target_model.predict(next_states, verbose=0)
            max_next_Q_values = np.max(next_Q_values, axis=1)

            # (1-done) yields 0 if done==1.0. If game is done, there is no useful next_state
            target_Q_values = rewards + (1.0 - done) * self.GAMMA * max_next_Q_values

            # compute gradients and apply
            with tf.GradientTape() as tape:  # automatic differentiation
                states_tf = tf.convert_to_tensor(states)
                allQvalues = self.model(states_tf, training=True)
                Qvalues = tf.gather(allQvalues, actions, batch_dims=1)
                target_tf = tf.convert_to_tensor(target_Q_values, dtype=tf.float32)
                loss = tf.reduce_mean(self.loss_fn(target_tf, Qvalues))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            
            if self.step % self.n_update_target_model == 0:
                self.update_target_model_weights()
                if self.step % 1000 == 0:
                    print(f"✓ Target Model aktualisiert (Step {self.step})")
            
            if self.step % 100 == 0:
                print(f"  Loss: {loss.numpy():.6f} | Q-values: min={Qvalues.numpy().min():.2f}, max={Qvalues.numpy().max():.2f}, mean={Qvalues.numpy().mean():.2f}")
            print(f"  Training... (Step {self.step}) | n_save_model: {self.n_save_model}")
            if self.step % self.n_save_model == 0:
                self.save_model()
            
            self.step = self.step + 1

    def save_model(self):
        """Speichert das trainierte Modell"""
        self.model.save(self.model_filename)
        print(f"✓ Modell gespeichert: {self.model_filename} (Step {self.step})")

    def get_action(self, state):
        """Wählt Aktion mit Epsilon-Greedy Strategie"""
        # Epsilon-greedy: exploration decision (decay controlled externally during training)
        if np.random.rand() <= self.EPSILON:
            return random.randrange(self.output_shape)
        q_values = self.model.predict(np.asarray(state)[np.newaxis], verbose=0)[0]
        return int(np.argmax(q_values))
    
    def update_target_model_weights(self):
        #update weights of target model with weights of trained model
        self.target_model.set_weights(self.model.get_weights())

