import random
import numpy as np
from const import *

def generate_history(Otraining):
    #build user history
    history=[]
    for i in range(Otraining.shape[0]):
        user_history=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0.0:
                user_history.append(j)
        random.shuffle(user_history)
        user_history=user_history[:HIS_LEN]    
        history.append(user_history+[Otraining.shape[1]+2]*(HIS_LEN-len(user_history)))
    history=np.array(history,dtype='int32')
    return history

def generate_training_data(Otraining,M):
    #build training user&items
    trainu=[]
    traini=[]
    trainlabel=[]
    train_user_index={}
    for i in range(Otraining.shape[0]):
        user_index=[]
        for j in range(len(Otraining[i])):
            if Otraining[i][j]!=0:
                user_index.append(len(trainu))
                trainu.append(i)
                traini.append(j)
                trainlabel.append(M[i][j]/LABEL_SCALE)
        if len(user_index):
            train_user_index[i]=user_index
            
    trainu=np.array(trainu,dtype='int32')
    traini=np.array(traini,dtype='int32')
    trainlabel=np.array(trainlabel,dtype='int32')
    return trainu,traini,trainlabel,train_user_index

def generate_test_data(Otest,M):
    #build test user&items
    testu=[]
    testi=[]
    testlabel=[]
    
    for i in range(Otest.shape[0]):
        for j in range(len(Otest[i])):
            if Otest[i][j]!=0:
                testu.append(i)
                testi.append(j)
                testlabel.append(M[i][j]/LABEL_SCALE)
    
    testu=np.array(testu,dtype='int32')
    testi=np.array(testi,dtype='int32')
    testlabel=np.array(testlabel,dtype='int32')
    return testu,testi,testlabel
