#import libraries
import os, importlib,sys
import numpy as np
import pandas as pd
import math
pd.options.mode.chained_assignment = None 
import matplotlib.pyplot as plt
#keras.models.Model.predict_proba = keras.models.Model.predict
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt

import pickle 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
import mplhep as hep
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight, resample
from joblib import dump, load

from sklearn.metrics import roc_curve, roc_auc_score 

import pickle 

import common_utils.plotting
from common_utils.plotting import plot_calibration_curve, plot_reweighted, plot_loss, fill_histograms_wError

from sklearn.utils import class_weight
from joblib import dump, load

from tensorflow.keras.models import model_from_json

class TrainEvaluatePreselNN:
    
    def __init__(self, dataset, columns, columns_scaling):
        
        self.dataset = dataset
        self.data_features_training = dataset[columns].copy()
        self.train_labels = dataset['train_labels_presel']
        self.weight_normed = dataset['weights_normed']
        self.columns = columns
        self.columns_scaling = columns_scaling

    # Defining a simple NN training for preselection - no need for "flexibility" here
    def train(self, test_size=0.15, random_state=42, path_to_save=''):

        # Split data into training and validation sets (including weights)
        X_train, X_val, y_train, y_val, weight_train, weight_val = train_test_split(self.data_features_training, 
                                                                                    self.train_labels, 
                                                                                    self.weight_normed, 
                                                                                    test_size=test_size, 
                                                                                    random_state=random_state, 
                                                                                    stratify=self.train_labels)

        # Standardize the input features
        # self.scaler = StandardScaler()
        self.scaler = ColumnTransformer([("scaler", MinMaxScaler(feature_range=(-1.5,1.5)), self.columns_scaling)],remainder='passthrough')
        X_train = self.scaler.fit_transform(X_train)  # Fit & transform training data
        X_val = self.scaler.transform(X_val)
        
        # Define the neural network model
        self.model = keras.Sequential([
            layers.Input(shape=(self.data_features_training.shape[1],)),  # Input layer
            layers.Dense(1000, activation='swish'),
            layers.Dense(1000, activation='swish'),
            layers.Dense(3, activation='softmax')  # Output layer for 5 classes
        ])

        optimizer = tf.keras.optimizers.Nadam(learning_rate=0.1)
        # Compile the model
        self.model.compile(optimizer='nadam',
                      loss='sparse_categorical_crossentropy',
                      weighted_metrics=["accuracy"])
        
        # Train the model with sample weights
        self.model.fit(X_train, y_train, sample_weight=weight_train, 
                  validation_data=(X_val, y_val, weight_val), epochs=20, batch_size=1024)

        if path_to_save!='':

            if not os.path.exists(path_to_save):
                os.makedirs(path_to_save)
    
            model_json = self.model.to_json()
            with open(path_to_save+"model_arch_presel.json", "w") as json_file:
                json_file.write(model_json)
    
            # serialize weights to HDF5
            self.model.save_weights(path_to_save+"model_weights_presel.weights.h5")

            saved_scaler = path_to_save+"model_scaler_presel.bin"
            dump(self.scaler, saved_scaler, compress=True)


    def get_trained_model(self, path_to_models):

        json_file = open(path_to_models+'/model_arch_presel.json', "r")

        loaded_model_json = json_file.read()

        json_file.close()

        self.model = model_from_json(loaded_model_json)

        self.model.load_weights(path_to_models+'/model_weights_presel.weights.h5')

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')

        self.scaler = load(path_to_models+'/model_scaler_presel.bin')


    def predict(self, dataset):

        features_scaled = self.scaler.transform(dataset[self.columns])
        pred_NN = self.model.predict(features_scaled)
        return pred_NN
    
        

class TrainEvaluate_NN:

    def __init__(self, dataset, weights, train_labels, columns, columns_scaling, 
                rnd_seed, sample_name, output_dir, output_name, path_to_figures='',
                     path_to_models='', path_to_ratios='',
                     use_log_loss=False, split_using_fold=False):

        self.dataset = dataset
        self.weights = weights
        self.train_labels = train_labels
        self.columns = columns
        self.columns_scaling = columns_scaling
        self.random_state_holdout = rnd_seed
        self.sample_name = sample_name
        self.output_dir = output_dir
        self.output_name = output_name
        self.use_log_loss = use_log_loss
        self.split_using_fold = split_using_fold
        
        self.path_to_figures = path_to_figures
        if not os.path.exists(path_to_figures):
                os.makedirs(path_to_figures)
            
        self.path_to_models = path_to_models
        if not os.path.exists(path_to_models):
                os.makedirs(path_to_models)

        self.path_to_ratios=path_to_ratios
        if not os.path.exists(path_to_ratios):
                os.makedirs(path_to_ratios)
        

    def train(self, hidden_layers, neurons, number_of_epochs, batch_size,
             learning_rate, scalerType, calibration=False, 
             num_bins_cal = 40, callback = True, 
             callback_patience=30, callback_factor=0.01,
             activation='swish'):

        self.calibration = calibration
        self.batch_size = batch_size

        #HyperParameters for the NN training
        holdout_num   = math.floor(self.dataset.shape[0]*0.2)
        train_num     = math.floor(self.dataset.shape[0]*0.8)
        validation_split = 0.1

        data_train, data_holdout, \
        label_train, label_holdout, \
        weight_train, weight_holdout = train_test_split(self.dataset[self.columns], 
                                                        self.train_labels, 
                                                        self.weights, 
                                                        test_size=holdout_num, 
                                                        random_state=self.random_state_holdout,
                                                        stratify=self.train_labels)


        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=callback_factor,
                                        patience=callback_patience, min_lr=0.000000001)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=300)


        if (scalerType == 'MinMax'):
            self.scaler = ColumnTransformer([("scaler",MinMaxScaler(feature_range=(-1.5,1.5)), self.columns_scaling)],remainder='passthrough')
        if (scalerType == 'StandardScaler'):
            self.scaler = ColumnTransformer([("scaler",StandardScaler(), self.columns_scaling)],remainder='passthrough')
        if (scalerType == 'PowerTransform_Yeo'):
            self.scaler = ColumnTransformer([("scaler",PowerTransformer(method='yeo-johnson', standardize=True), self.columns_scaling)],remainder='passthrough')
        if (scalerType == 'RobustScaler'):
            self.scaler = ColumnTransformer([("scaler",RobustScaler(unit_variance=True), self.columns_scaling)],remainder='passthrough')
        if (scalerType == 'QuantileTransformer'):
            self.scaler = ColumnTransformer([("scaler",QuantileTransformer(output_distribution='normal'), self.columns_scaling)],remainder='passthrough')


        scaled_data_train = self.scaler.fit_transform(data_train)
        scaled_data_train= pd.DataFrame(scaled_data_train, columns=self.columns)

        scaled_data_holdout = self.scaler.transform(data_holdout)
        scaled_data_holdout = pd.DataFrame(scaled_data_holdout, columns=self.columns)

        # Check if the datasets are normalized
        print(f"Sum of weights of class 0: {np.sum(weight_train[label_train==0])}")
        print(f"Sum of weights of class 1: {np.sum(weight_train[label_train==1])}")

        print(f"Using {activation} activation function")

        self.model_NN = build_model(n_hidden=hidden_layers, n_neurons=neurons, 
                                    learning_rate=learning_rate, 
                                    input_shape=[len(self.columns)], 
                                    use_log_loss=self.use_log_loss,
                                    activation=activation)

        self.model_NN.summary()

        if callback:

            print("Using Callbacks")

            self.history = self.model_NN.fit(scaled_data_train, label_train, callbacks=[reduce_lr, es], 
                                                epochs=number_of_epochs, batch_size=batch_size, 
                                                validation_split=validation_split, sample_weight=weight_train, 
                                                verbose=2)

        else:
            print("Not Using Callbacks")

            self.history = self.model_NN.fit(scaled_data_train, label_train, 
                                                epochs=number_of_epochs, batch_size=batch_size, 
                                                validation_split=validation_split, sample_weight=weight_train, 
                                                verbose=2)
        

        print("Finished Training")


        saved_scaler = self.path_to_models+"model_scaler.bin"
        print(saved_scaler)

        model_json = self.model_NN.to_json()
        with open(self.path_to_models+"model_arch.json", "w") as json_file:
            json_file.write(model_json)

        # serialize weights to HDF5
        self.model_NN.save_weights(self.path_to_models+"model_weights.weights.h5")

        dump(self.scaler, saved_scaler, compress=True)

        plot_loss(self.history, path_to_figures=self.path_to_figures)

        # Redo the split with all the columns in the original dataset
        self.train_data_eval, self.holdout_data_eval = train_test_split(self.dataset, 
                                                                        test_size=holdout_num, 
                                                                        random_state=self.random_state_holdout,
                                                                        stratify=self.train_labels)
        

        holdout_data_prediction = self.predict_with_model(self.holdout_data_eval, use_log_loss=self.use_log_loss)

        self.label_0_hpred = holdout_data_prediction[label_holdout==0].copy()
        self.label_1_hpred = holdout_data_prediction[label_holdout==1].copy()

        self.w_holdout_label_0 = weight_holdout[label_holdout==0].copy()
        self.w_holdout_label_1 = weight_holdout[label_holdout==1].copy()

        train_data_prediction = self.predict_with_model(self.train_data_eval, use_log_loss=self.use_log_loss)

        self.label_0_tpred = train_data_prediction[label_train==0].copy()
        self.label_1_tpred = train_data_prediction[label_train==1].copy()
        
        self.w_train_label_0 = weight_train[label_train==0].copy()
        self.w_train_label_1 = weight_train[label_train==1].copy()

        
        # Some diagnostics to ensure numerical stability - min/max must not be exactly 0 or 1
        print(f"{self.sample_name[1]} training data prediction (max) = "+str(np.amax(self.label_0_tpred)))
        print(f"{self.sample_name[1]} training data prediction (min) = "+str(np.amin(self.label_0_tpred)))

        print(f"{self.sample_name[0]} training data prediction (max) = "+str(np.amax(self.label_1_tpred)))
        print(f"{self.sample_name[0]} training data prediction (min) = "+str(np.amin(self.label_1_tpred)))

        print(f"{self.sample_name[1]} holdout data prediction (max) = "+str(np.amax(self.label_0_hpred)))
        print(f"{self.sample_name[1]} holdout data prediction (min) = "+str(np.amin(self.label_0_hpred)))

        print(f"{self.sample_name[0]} holdout data prediction (max) = "+str(np.amax(self.label_1_hpred)))
        print(f"{self.sample_name[0]} holdout data prediction (min) = "+str(np.amin(self.label_1_hpred)))



    def get_trained_model(self, path_to_models, calibration = False):

        self.calibration = calibration

        json_file = open(path_to_models+'/model_arch.json', "r")

        loaded_model_json = json_file.read()

        json_file.close()

        self.model_NN = model_from_json(loaded_model_json)

        self.model_NN.load_weights(path_to_models+'/model_weights.h5')

        opt = tf.keras.optimizers.Nadam(learning_rate=0.1)

        self.model_NN.compile(loss='binary_crossentropy', optimizer=opt)

        self.scaler = load(path_to_models+'/model_scaler.bin')

    
    def predict_with_model(self, data, use_log_loss=False):

        scaled_data = self.scaler.transform(data)
        pred = self.model_NN.predict(scaled_data, verbose=2)
        pred = pred.reshape(pred.shape[0],)

        if use_log_loss:

            pred = convert_to_score(pred)

        if self.calibration:

            pred = self.calibrator_obj.cali_pred(pred)
            pred = pred.reshape(pred.shape[0],)

        return pred
        

    def make_overfit_plots(self):

        importlib.reload(sys.modules['common_utils.plotting'])
        from common_utils.plotting import plot_overfit

        plot_overfit(self.label_0_tpred, self.label_0_hpred, self.w_train_label_0, self.w_holdout_label_0, 
                    calibration_frac = 0.3, nbins=30, plotRange=[0.0,1.0], holdout_index=0, 
                    label=f'{self.sample_name[0]}', path_to_figures=self.path_to_figures)
        
        plot_overfit(self.label_1_tpred, self.label_1_hpred, self.w_train_label_1, self.w_holdout_label_1,
                    calibration_frac = 0.3, nbins=30, plotRange=[0.0,1.0], holdout_index=0, 
                    label=f'{self.sample_name[1]}', path_to_figures=self.path_to_figures)


    def make_calib_plots(self, observable='score', nbins=10):

        importlib.reload(sys.modules['common_utils.plotting'])
        from common_utils.plotting import plot_calibration_curve, plot_calibration_curve_ratio

        if observable=='score':
            # Plot Calibration curves - score function
            plot_calibration_curve(self.label_0_tpred, self.w_train_label_0, 
                                   self.label_1_tpred, self.w_train_label_1, 
                                   self.label_0_hpred, self.w_holdout_label_0, 
                                   self.label_1_hpred, self.w_holdout_label_1, 
                                   self.path_to_figures, nbins=nbins, 
                                   label="Calibration Curve - "+str(self.sample_name[0]))

        # Plot Calibration curves - nll function
        if observable=='llr':
        
            plot_calibration_curve_ratio(self.label_0_tpred, self.w_train_label_0, 
                                         self.label_1_tpred, self.w_train_label_1, 
                                         self.label_0_hpred, self.w_holdout_label_0, 
                                         self.label_1_hpred, self.w_holdout_label_1, 
                                         self.path_to_figures, nbins=nbins, 
                                         label="Calibration Curve - "+str(self.sample_name[0]))

        else:
            print("observable not recognized")


    def make_reweighted_plots(self, variables, scale, num_bins):

        importlib.reload(sys.modules['common_utils.plotting'])
        from common_utils.plotting import plot_reweighted

        plot_reweighted(self.train_data_eval, self.label_0_tpred, 
                        self.w_train_label_0, self.label_1_tpred, self.w_train_label_1,
                        variables=variables, num=num_bins,
                        sample_name=self.sample_name, scale=scale,  
                        path_to_figures=self.path_to_figures, label='Training Data Diagnostic')

        plot_reweighted(self.holdout_data_eval, self.label_0_hpred, 
                        self.w_holdout_label_0,self.label_1_hpred, self.w_holdout_label_1,
                        variables=variables, num=num_bins,
                        sample_name=self.sample_name, scale=scale, 
                        path_to_figures=self.path_to_figures, label='Holdout Data Diagnostic')


    def evaluate_and_save_ratios(self, dataset):

        channel_name = self.sample_name[0]
        
        score_pred = self.predict_with_model(dataset[self.columns], use_log_loss=self.use_log_loss)

        ratio = score_pred/(1.0-score_pred)

        path_to_ratios=f'{self.output_dir}/ratios_{self.output_name}/'

        np.save(f"{self.path_to_ratios}ratio_{self.sample_name[0]}_bs{self.batch_size}rnd{str(self.random_state_holdout)}.npy", ratio)


def fill_histograms_wError(data, weights, edges, histrange, epsilon, normalize=True):
        
    h, _ = np.histogram(data, edges, histrange, weights=weights)
    
    if normalize:
        
        i = np.sum(h)

        h = h/i

    h_err, _ = np.histogram(data, edges, histrange, weights=weights**2)

    if normalize:
        
        h_err = h_err/(i**2)
    
    return h, h_err


def build_model(n_hidden=4, n_neurons=1000, learning_rate=0.1, 
                input_shape=[11], use_log_loss=False, optimizer_choice='Nadam', 
                activation='swish'):
    
    model = tf.keras.models.Sequential()
    options = {"input_shape":input_shape}
    for layer in range(n_hidden):

        if activation=='mish':
            def mish(inputs):
                x = tf.nn.softplus(inputs)
                x = tf.nn.tanh(x)
                x = tf.multiply(x, inputs)
                return x

            model.add(Dense(n_neurons, activation=mish, **options))
        else:
            model.add(Dense(n_neurons, activation=activation, **options))
        options={}

    if not use_log_loss:
        model.add(Dense(1,activation='sigmoid',**options))
    else:
        model.add(Dense(1,activation='linear',**options))

    if optimizer_choice=='Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate) 
    elif optimizer_choice=='Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 

    if use_log_loss:
        model.compile(loss=CARL_mod, optimizer=optimizer, weighted_metrics=['binary_accuracy'])
    else:
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer, weighted_metrics=['binary_accuracy'])
    return model

#The modified CARL loss that directly regresses to LLR
def CARL_mod(s_truth, s_predicted): 

    loss = (s_truth)*tf.exp(-0.5*s_predicted)+(1.0-s_truth)*tf.exp(0.5*s_predicted)
    return loss
