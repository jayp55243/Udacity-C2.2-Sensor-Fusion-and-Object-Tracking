# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import logging
# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        #time delta
        self.dt = params.dt
        
        #design parameter for process noise
        self.q = params.q
        
        #dimension for process model
        self.dim_state = params.dim_state


    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        
        #dimension for process model
        n = self.dim_state
        
        #initilize system matrix
        F = np.eye(n)
        
        #Update system matrix
        F[0, 3] = F[1, 4] = F[2, 5] = self.dt 
 
        return np.matrix(F)
        
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        #dimension for process model, time delta, process noise design parameter
        n = self.dim_state
        dt = self.dt
        q = self.q
        
        #Initialize process covariance
        Q = np.zeros((n, n))
        
        #update process covariance
        Q[0, 0] = Q[1, 1] = Q[2, 2] = (dt**3)/3*q
        Q[0, 3] = Q[1, 4] = Q[2, 5] = Q[3, 0] = Q[4, 1] = Q[5, 2] = (dt**2)/2*q
        Q[3, 3] = Q[4, 4] = Q[5, 5] =  dt*q
        
        return np.matrix(Q)
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        #initialize sytem and process covariance matrices
        F = self.F()
        Q = self.Q()
        
        # state prediction
        x = F * track.x 
        
        # covariance prediction
        P = F * track.P * F.transpose() + Q 
        
        #Save x, P in track
        track.set_x(x)
        track.set_P(P)
        
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        
        #dimension for process model
        n = self.dim_state
        
        # measurement matrix
        H = meas.sensor.get_H(track.x) 
        
        # residual
        gamma = self.gamma(track, meas) 
        
        # covariance of residual
        S = self.S(track, meas, H) 
        
        # Kalman gain
        K = track.P * H.transpose()* np.linalg.inv(S)
        
        # state update
        x = track.x + K * gamma 
        
        #Initialize Identity matrix of size n
        I = np.identity(n)
        
        # covariance update
        P = (I - K * H) * track.P 
        
        #Save x, P in track
        track.set_x(x)
        track.set_P(P)
            
        
            
        ############
        # END student code
        ############ 
        track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############    
        #Calc residual gamma
        gamma = meas.z - meas.sensor.get_hx(track.x)
        
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        #Calc covariance of residual S
        S = H * track.P * H.transpose() + meas.R 
        
        return S
        
        ############
        # END student code
        ############ 