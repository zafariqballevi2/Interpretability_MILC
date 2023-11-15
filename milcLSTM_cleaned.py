'''
--- This code is intended to create a cleaned version of milc architecture using LSTM encoder 
--- Aimed to work on synthetic data
--- To share with Zafar Iqbal 

Contributor: Rahman, M. M, TReNDS center
Usage:

To train model, use: python milcLSTM_cleaned.py gaid_id 'train', e.g. to train a model with gain value=1.0, use: python milcLSTM_cleaned.py 10 'train'
To compute saliency, use: python milcLSTM_cleaned.py gain_id 'attribution'
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

# from source.utils import get_argparser
# from source.encoders_ICA import NatureCNN, ImpalaCNN, NatureOneCNN, LinearEncoder

from datetime import datetime

import sys
import os
import time

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    Occlusion, 
    Saliency,
    GuidedBackprop,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Gain = {1:0.1, 2:0.2, 3:0.3, 4:0.4, 5:0.5, 6:0.6, 7:0.7, 8:0.8, 9:0.9, 10:1.0, 11:1.1, 12:1.2, 13:1.3, 14:1.4, 15:1.5, 16:1.6, 17:1.7, 18:1.8, 19:1.9, 20:2.0}  

# set gain key , use 1, 2, 3, 4, 5, or any key from the dict you like to set
ID = int(sys.argv[1])

# want to train or compute attribution?
train_or_attribution = str(sys.argv[2])

# ensemble key 
# Ensembles = {0:'', 1:'smoothgrad', 2:'smoothgrad_sq', 3: 'vargrad'}
# FilesDic = {'smoothgrad': 'SG' , 'smoothgrad_sq': 'SGSQ', 'vargrad': 'VG', '':''}
# EID = int(sys.argv[2])


class LSTM(torch.nn.Module):

    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, gain):
        super(LSTM, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = hidden_nodes
        
        self.enc_out = input_size
        self.lstm = nn.LSTM(input_size, hidden_nodes, batch_first=True)
        
        # input size for the top lstm is the hidden size for the lower
        
        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)  

        # previously, I used 64
        self.attnenc = nn.Sequential(
             nn.Linear(self.enc_out, 64),
             nn.Linear(64, 1)
        )
     
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.Linear(128, 1)
        )
        
        # Previously it was 64, now used 200
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 200),
            nn.Linear(200, output_size)
        )
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing fresh components')
        for name, param in self.encoder.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attnenc.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.decoder.named_parameters():
            print('Initializing Decoder:', name)
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        for name, param in self.attn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param, gain=self.gain)

    def forward(self, x):
        
        b_size = x.size(0)
        s_size = x.size(1)
        x = x.view(-1, x.shape[2], 20)
        x = x.permute(0, 2, 1)
        
        
        out, hidden = self.encoder(x)
        out = self.get_attention_enc(out)
        out = out.view(b_size, s_size, -1)
        lstm_out, hidden = self.lstm(out)
        

        # lstm_out = self.lstm(x)

        lstm_out = self.get_attention(lstm_out)
        
        lstm_out = lstm_out.view(b_size, -1)

        smax = torch.nn.Softmax(dim=1)
        lstm_out_smax = smax(lstm_out)
       
        return lstm_out #lstm_out_smax    #lstm_out_smax

    def get_attention(self, outputs):
        
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs2.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.hidden)

        weights = self.attn(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()

        # Pass the weighted output to decoder
        logits = self.decoder(attn_applied)
        return logits
    
    def get_attention_enc(self, outputs):
        
        # For anchor point
#         B= outputs[:,-1, :]
#         B = B.unsqueeze(1).expand_as(outputs)
#         outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs.reshape(-1, self.enc_out)

        weights = self.attnenc(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()

        return attn_applied


class DataWithLabels(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.Y[i]


def get_data_loader(X, Y, batch_size, shuffle=False):
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle)

    return dataLoader


def train_model(model, loader_train, loader_test, epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model.cuda()
    model.to(device)

    for epoch in range(epochs):

        for i, data in enumerate(loader_train):
            x, y = data
            # x = x.permute(1, 0, 2)
            optimizer.zero_grad()
            outputs = model(x)
           
            l = loss(outputs, y)
            _, preds = torch.max(outputs.data, 1)
            accuracy = (preds == y).sum().item()
            l.backward()

            optimizer.step()
        x_test, y_test = next(iter(loader_test))
        # x_test = x_test.permute(1, 0, 2)
        outputs = model(x_test)
        
        _, preds = torch.max(outputs.data, 1)
        accuracy = (preds == y_test).sum().item()
        print("epoch: " + str(epoch) + ", loss: " + str(l.detach().item()) +", train acc: " + str(accuracy / y_test.size(0)))
    return optimizer


def get_captum_saliency(model, loaderSal):
    # model.eval()
    model.zero_grad()
    for param in model.parameters():
        param.requires_grad = False
    sal = IntegratedGradients(model)
    
    # nt = NoiseTunnel(sal)
    
    saliencies = []
    for i, data in enumerate(loaderSal):
        if i % 1000 == 0:
            print(i)
        x, y = data   
        bl = torch.zeros(x.shape).to(device)
        x = x.to(device)
        y = y.to(device)
        x.requires_grad_()
        
        S = sal.attribute(x,bl,target=y)


        # nt_type options: `smoothgrad`, `smoothgrad_sq` or `vargrad`
        # S = nt.attribute(x, nt_type=Ensembles[EID], n_samples=10, baselines = bl, target=y)

        saliencies.append(np.squeeze(S.cpu().detach().numpy()))
       
        
    return saliencies


def artificial_batching_patterned_space2(samples, t_steps, features, p_steps=10, seed=None):
    # we used this in our experiemnt
    print('Old two class data...')
    if seed != None:
        np.random.seed(seed)
    X = np.zeros([samples, t_steps, features])
    L = np.zeros([samples])
    start_positions = np.zeros([samples])
    masks = np.zeros([samples,p_steps,features])
    for i in range(samples):
        mask = np.zeros([p_steps, features])
        #0,17 ; 27,47
        start = np.random.randint(0,t_steps-p_steps)
        start_positions[i] = start
        x = np.random.normal(0, 1, [1, t_steps, features])
        label = np.random.randint(0, 2)
        lift = np.random.normal(1, 1,[p_steps,features])#np.random.normal(0, 1, [p_steps, features])
        X[i,:,:] = x
        if label:
            mask[:,0:int(features/2)] = 1
        else:
            mask[:,int(features/2):] = 1
        lift = lift*mask
        X[i,start:start+p_steps, :] += lift
        masks[i,:,:] = lift
        L[i] = int(label)
    return X, L, start_positions, masks


X, Y, start_positions, masks = artificial_batching_patterned_space2(3000, 140, 50, seed=1988)

X = np.moveaxis(X, 1, 2)  # it needs only for encoder

print('Original Data Shape:', X.shape)

subjects = 3000
sample_x = X.shape[1] #A.shape[0]
sample_y = 20
tc = 140
samples_per_subject = 121 #7 #7 #7 #121
window_shift = 1 #20

finalData = np.zeros((subjects, samples_per_subject, sample_x, sample_y))

for i in range(subjects):
    for j in range(samples_per_subject):
        finalData[i, j, :, :] = X[i, :, (j * window_shift):(j * window_shift) + sample_y]
        
X = finalData

print('Data shape ended up with:', X.shape)

# train samples
X_train = X[:2000]
Y_train = Y[:2000] 

X_train = torch.from_numpy(X_train).float().to(device)
Y_train = torch.from_numpy(Y_train).long().to(device)

# test samples
X_test = X[2500:]
Y_test = Y[2500:]
              

# saliency samples   
X_sal = X[2000:2100]
Y_sal = Y[2000:2100]
              
X_test = torch.from_numpy(X_test).float().to(device)
Y_test = torch.from_numpy(Y_test).long().to(device)

X_sal = torch.from_numpy(X_sal).float().to(device)
Y_sal = torch.from_numpy(Y_sal).long().to(device)

dataLoaderTrain = get_data_loader(X_train, Y_train, 64)
dataLoaderTest = get_data_loader(X_test, Y_test, X_test.shape[0])

# create path/directories
MainDir = "Stride1Dir"
sub_dir = "NPT"
project_dir = 'Zafar_Time_Reversal_Project'
wpath = os.path.join('./', project_dir)
main_path = os.path.join(wpath, MainDir, sub_dir)

if not os.path.exists(main_path):
    os.makedirs(main_path)


gain_value = Gain[ID]

for restart in range(1, 2, 1):
    
    start_time = time.time()
    print('Model {} started'.format(restart))
    
    model = LSTM(X.shape[2], 256, 200, 121, 2, gain_value).float()
    
    
    print('MILC + with TOP Anchor + both uniLSTM')
    print('Gain value used:', gain_value)

    # set a prefix to name files

    prefix = "top_down_data_lstm_only_gain_"+str(gain_value)+"_"

    if train_or_attribution == 'train':

        # train the model 
        optimizer = train_model(model, dataLoaderTrain, dataLoaderTest, 100, .001)

        # Save model

        model_path = os.path.join(main_path, prefix+'un_softmaxed_captum_use_'+str(restart)+ '.pt')
        torch.save(model.state_dict(), model_path)
        print('Model saved here:', model_path)

    elif train_or_attribution == 'attribution':
        dataLoaderSal = get_data_loader(X_sal, Y_sal, 1)
        print('Loading model for saliency computation...')
        model_path = os.path.join(main_path, prefix+'un_softmaxed_captum_use_'+str(restart)+ '.pt')

        # re-load the model
        
        model_dict = torch.load(model_path, map_location=device) 
        model.load_state_dict(model_dict)
        model.to(device)


        saliencies = get_captum_saliency(model, dataLoaderSal)
        saliencies1 = np.stack(saliencies, axis=0)
        print(saliencies1.shape)

        # saliency directory 
        sal_directory = os.path.join(main_path, 'Saliency')
        if not os.path.exists(sal_directory):
            os.makedirs(sal_directory)

        sal_file_path = os.path.join(sal_directory, prefix+'ep100_bs_64_ig_attribution_'+str(restart))
        print('Saliency saved here:', sal_file_path)

        np.save(sal_file_path, saliencies1)

    else:
        print('2nd argument wrong. Try again with the correct argument.')
  
    elapsed_time = time.time() - start_time
    print('Total Time Elapsed:', elapsed_time)
   
    
print(datetime.now())