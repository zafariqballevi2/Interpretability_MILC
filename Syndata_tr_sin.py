import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable
from pathlib import Path
import pickle
import mat73
from sklearn.model_selection import StratifiedKFold, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
import pandas as pd
import sys
import os
import time
import gc
from pathlib import Path
import random

from wholeMILC import NatureOneCNN, Flatten
from lstm_attn import subjLSTM
from All_Architecture import combinedModel
from utils import get_argparser
parser = get_argparser()
args = parser.parse_args()
print(args.jobid)
device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
print(device)

Trials = 1
eppochs = 200
seeds = 1
Gain = [0.9]
#Gain = [1.2, 0.9, 0.7, 0.65]
enc = [ 'cnn', 'lstmM']

start_time = time.time()


# np.random.seed(run)

def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


class LSTM(torch.nn.Module):

    def __init__(self, enc_input_size, input_size, hidden_nodes, sequence_size, output_size, gain):
        super(LSTM, self).__init__()
        self.sequence_size = sequence_size
        self.hidden = hidden_nodes
        
        self.enc_out = input_size
        self.lstm = nn.LSTM(input_size, hidden_nodes, batch_first=True)
        
        # input size for the top lstm is the hidden size for the lower
        
        self.encoder = nn.LSTM(enc_input_size, self.enc_out, batch_first = True)  

        self.attnenc = nn.Sequential(
             nn.Linear(2*self.enc_out, 64),
             nn.ReLU(),
             nn.Linear(64, 1)
        )
     
        self.attn = nn.Sequential(
            nn.Linear(2*self.hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, 200),
            nn.ReLU(),
            nn.Linear(200, output_size),
            nn.Sigmoid()
        )
        
        self.gain = gain
        
        self.init_weight()

        
    def init_weight(self):
        
        # For UFPT experiments, initialize only the decoder
        print('Initializing All components')
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
                
    def init_hidden(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden, device=device))
        
        return (h0, c0)
    
    def init_hidden_enc(self, batch_size, device):
        
        h0 = Variable(torch.zeros(1, batch_size, self.enc_out, device=device))
        c0 = Variable(torch.zeros(1, batch_size, self.enc_out, device=device))
        
        return (h0, c0)


    def forward(self, x):
        
        sx = []
        for episode in x:
            mean = episode.mean()
            sd = episode.std()
            episode = (episode - mean) / sd
            sx.append(episode)

        x = torch.stack(sx)
        
        b_size = x.size(0)
        s_size = x.size(1)
        x = x.view(-1, x.shape[2], 20)
        x = x.permute(0, 2, 1)
        
        enc_batch_size = x.size(0)
            
        self.enc_hidden = self.init_hidden_enc(enc_batch_size, device)
        out, self.enc_hidden = self.encoder(x, self.enc_hidden)

        out = self.get_attention_enc(out)
        out = out.view(b_size, s_size, -1)

        self.lstm_hidden = self.init_hidden(b_size, device)
        lstm_out, self.lstm_hidden = self.lstm(out, self.lstm_hidden)
        
        
#         out, hidden = self.encoder(x)
#         out = self.get_attention_enc(out)
#         out = out.view(b_size, s_size, -1)
#         lstm_out, hidden = self.lstm(out)
        

        # lstm_out = self.lstm(x)

        lstm_out = self.get_attention(lstm_out)
        
        lstm_out = lstm_out.view(b_size, -1)
        
        smax = torch.nn.Softmax(dim=1)
        lstm_out_smax = smax(lstm_out)
        
        return lstm_out  #lstm_out_smax

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
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2*self.enc_out)

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


def get_data_loader(X, Y, batch_size):
    dataLoader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle = True)

    return dataLoader


def train_model(model, loader_train, loader_train_check, loader_test, epochs, learning_rate):
    loss = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # model.cuda()
    model.to(device)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):

        for i, data in enumerate(loader_train):
            x, y = data
            # x = x.permute(1, 0, 2)
            optimizer.zero_grad()
            outputs= model(x)
           
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
        
        sig = F.softmax(outputs, dim=1).to(device)
        y_scores = sig.detach()[:, 1]
        roc = roc_auc_score(y_test.to('cpu'), y_scores.to('cpu'))
        
        
        x_train, y_train = next(iter(loader_train_check))

        train_outputs = model(x_train)
        _, train_preds = torch.max(train_outputs.data, 1)
        train_accuracy = (train_preds == y_train).sum().item()
        train_accuracy /= y_train.size(0)
        
        train_sig = F.softmax(train_outputs, dim=1).to(device)
        y_train_scores = train_sig.detach()[:, 1]
        train_roc = roc_auc_score(y_train.to('cpu'), y_train_scores.to('cpu'))
        
        print("epoch: " + str(epoch) + ", loss: " + str(l.detach().item()) +", test acc: " + str(accuracy / y_test.size(0)) + ", roc: " + str(roc) +", train acc: " + str(train_accuracy) +" , train roc: " + str(train_roc))
        
        test_loss = loss(outputs, y_test)
        scheduler.step(test_loss)
        
        
    return optimizer, accuracy / y_test.size(0), roc, model


            
print(torch.cuda.is_available())


MODELS = {0: 'FPT', 1: 'UFPT', 2: 'NPT'}
Dataset = {0: 'SYNDATA', 1: 'COBRE', 2: 'OASIS', 3: 'ABIDE'}
# gain = {'COBRE': [0.05, 0.65, 0.75], 'FBIRN': [0.85, 0.4, 0.35], 'ABIDE': [0.3, 0.35, 0.8],
#         'OASIS': [0.4, 0.65, 0.35]}

#Directories = { 'COBRE': 'COBRESaliencies','FBIRN' : 'FBIRNSaliencies', 'ABIDE' : 'ABIDESaliencies','OASIS' : 'OASISSaliencies'}

#FNCDict = {"FBIRN": LoadFBIRN, "COBRE": LoadCOBRE, "OASIS": LoadOASIS, "ABIDE": LoadABIDE}

Params = {'SYNDATA':[2000, 140, 121], 'COBRE': [157, 140, 121], 'OASIS': [372, 120, 101], 'ABIDE': [569, 140, 121]}

#train_lim = { 'FBIRN': 250, 'COBRE': 100, 'OASIS': 300, 'ABIDE': 400}

#Params_subj_distr = { 'FBIRN': [100,100,100,100,100, 100, 100, 100], 'COBRE': [15, 25, 40], 'OASIS': [15, 25, 50, 75, 100, 120], 'ABIDE': [15, 25, 50, 75, 100, 150]}
Params_subj_distr = { 'SYNDATA': [980, 980, 980, 980, 980, 980, 980, 980], 'COBRE': [15, 25, 40], 'OASIS': [15, 25, 50, 75, 100, 120], 'ABIDE': [15, 25, 50, 75, 100, 150]}
#[15, 25, 50, 75, 100]
test_lim_per_class = { 'SYNDATA': 32, 'COBRE': 16, 'OASIS': 32, 'ABIDE': 50}

# These gains were based on cross-validation based search

# Params_best_gains = {1: {'FBIRN': {'NPT':[1.0, 0.9, 0.9, 1.0, 1.0], 'UFPT': [1.0, 0.8, 0.7, 0.5, 0.4]}, 'OASIS': {'NPT': [0.9, 0.9, 1.0, 1.1, 1.1, 1.0], 'UFPT': [0.8, 0.8, 0.4, 0.3, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.0, 0.9, 0.9, 1.0, 0.7, 0.6], 'UFPT': [0.4, 0.8, 0.3, 0.2, 0.2, 0.1]}, 'COBRE': {'NPT': [1.1, 0.7, 0.8], 'UFPT': [1.6, 1.3, 1.3]}}, 
#                      10: {'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [1.2, 1.2, 1.4]}},
#                      20: {'COBRE': {'NPT': [1.5, 0.8, 0.8], 'UFPT': [1.2, 1.3, 1.3]}}  
#                     }

# Data normalize and cross-validation search (cross validation on training for FBIRN)
# simple gain selection (used only one fold for ABIDE) 

# Params_best_gains = {1: {'FBIRN': {'NPT':[0.9, 0.9, 0.8, 0.7, 0.6], 'UFPT': [0.8, 0.8, 0.6, 0.7, 0.6]}, 'ABIDE': {'NPT': [0.9, 0.5, 0.9, 1.0, 1.0, 0.7], 'UFPT': [1.0, 0.9, 0.4, 0.4, 0.1, 0.2]}, 'OASIS': {'NPT': [0.6, 0.9, 1.0, 0.7, 0.9, 0.6], 'UFPT': [0.5, 0.4, 0.3, 0.6, 1.0, 1.1]},'COBRE': {'NPT': [1.0, 0.3, 0.5], 'UFPT': [0.7, 0.6, 1.2]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}

# Revised h_fixed Simiple Gain Selection

#Params_best_gains = {1: {'FBIRN': {'NPT':[1.2, 1.3, 0.7, 1.1, 0.9], 'UFPT': [0.9, 0.7, 0.6, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.3, 1.3, 1.1, 0.9, 1.3, 0.6], 'UFPT': [0.4, 0.6, 0.5, 0.5, 0.3, 1.2]}, 'OASIS': {'NPT': [0.1, 1.2, 1.0, 1.0, 1.2, 0.7], 'UFPT': [0.7, 0.7, 1.3, 0.6, 1.1, 1.2]},'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [0.4, 1.1, 1.3]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}
#Params_best_gains = {1: {'SYNDATA': {'NPT':[0.9, 0.7, 0.6, 0.5, 0.4, 0.05, 0.45, 0.65], 'UFPT': [0.9, 0.7, 0.6, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.3, 1.3, 1.1, 0.9, 1.3, 0.6], 'UFPT': [0.4, 0.6, 0.5, 0.5, 0.3, 1.2]}, 'OASIS': {'NPT': [0.1, 1.2, 1.0, 1.0, 1.2, 0.7], 'UFPT': [0.7, 0.7, 1.3, 0.6, 1.1, 1.2]},'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [0.4, 1.1, 1.3]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}
#Params_best_gains = {1: {'SYNDATA': {'NPT':[1.2, 0.9], 'UFPT': [0.9, 0.7, 0.6, 0.5, 0.4]}, 'ABIDE': {'NPT': [1.3, 1.3, 1.1, 0.9, 1.3, 0.6], 'UFPT': [0.4, 0.6, 0.5, 0.5, 0.3, 1.2]}, 'OASIS': {'NPT': [0.1, 1.2, 1.0, 1.0, 1.2, 0.7], 'UFPT': [0.7, 0.7, 1.3, 0.6, 1.1, 1.2]},'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [0.4, 1.1, 1.3]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}
#Params_best_gains = {1: {'SYNDATA': {'NPT':[0.65,0.65,0.65, 0.65,0.65], 'UFPT': [0.65,0.65,0.65, 0.65,0.65]}, 'ABIDE': {'NPT': [1.3, 1.3, 1.1, 0.9, 1.3, 0.6], 'UFPT': [0.4, 0.6, 0.5, 0.5, 0.3, 1.2]}, 'OASIS': {'NPT': [0.1, 1.2, 1.0, 1.0, 1.2, 0.7], 'UFPT': [0.7, 0.7, 1.3, 0.6, 1.1, 1.2]},'COBRE': {'NPT': [0.9, 0.9, 1.0], 'UFPT': [0.4, 1.1, 1.3]}}, 20: {'COBRE': {'NPT': [0.6, 1.1, 1.1], 'UFPT': [1.1, 1.5, 0.6]}}}
############

path_a = '/data/users2/ziqbal5/abc/MILC_LSTM/Data/'
def generate_sample_synthetic_data(subjects, tp, components):

    data = np.empty([subjects, components,tp])
    for j in range(subjects):
        for i in range(components):

            # Parameters
            duration = 5.0  # seconds
            start_freq =  random.uniform(5,10)  # Hz0
            end_freq =  random.uniform(20,25)   # Hz
            sampling_rate = 208  # Hz

            # Time array
            t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

            # Phase array
            phase = 2 * np.pi * (start_freq * t + 0.5 * (end_freq - start_freq) * t ** 2 / duration)

            # Signal (sine wave with linearly increasing frequency)
            data[j][i] = np.sin(phase)

        #Forward Direction
        X1 = torch.Tensor(data)
        L1 = np.zeros([len(data),])
        L1 = torch.tensor(L1)
        
        split = int(subjects*.80)
        tr_data1 = X1[:split]
        tr_labels1 = L1[:split]

        test_data1 = X1[split:]
        test_labels1 = L1[split:]

        #Reverse Direction
        ccc = np.empty([subjects, components, tp])
        for i in range(subjects):
            for j in range(components):
                ccc[i][j] = np.flip(data[i][j])
        print("Slices reversal is done")

        X2 = torch.Tensor(ccc)
        L2 = np.ones([len(ccc),])
        L2 = torch.tensor(L2)
        
    
        tr_data2 = X2[:split]
        tr_labels2 = L2[:split]

        test_data2 = X2[split:]
        test_labels2 = L2[split:]

        tr_data = torch.cat((tr_data1, tr_data2), 0)
        tr_labels = torch.cat((tr_labels1, tr_labels2),0)

        test_data = torch.cat((test_data1, test_data2),0)
        test_labels = torch.cat((test_labels1, test_labels2),0)
                
        # with open(os.path.join(path_a, str(args.jobid) +'_' + str(dd) + 'tr_data.pickle'), "wb") as outfile:
        #     pickle.dump(tr_data, outfile)
        # with open(os.path.join(path_a, str(args.jobid) +'_' +str(dd) + 'tr_labels.pickle'), "wb") as outfile:
        #     pickle.dump(tr_labels, outfile)
        # with open(os.path.join(path_a, str(args.jobid) +'_' +str(dd) + 'test_data.pickle'), "wb") as outfile:
        #     pickle.dump(test_data, outfile)
        # with open(os.path.join(path_a, str(args.jobid) +'_' +str(dd) + 'test_labels.pickle'), "wb") as outfile:
        #     pickle.dump(test_labels, outfile)
          

        return tr_data, tr_labels, test_data, test_labels   

for dd in range(seeds):
    print("Seed: ", dd)
    tr_data, tr_labels, _, _ = generate_sample_synthetic_data(1250, 1040, 53)
    print(tr_data.shape, tr_labels.shape, len(tr_data))
    for en in range(len(enc)):
        encoderr = enc[en]
        print("Encoder: ", encoderr)
        data = 0
        mode = 2

        finalData = tr_data
        all_labels = tr_labels


        no_good_comp = 53
        sample_y = 20
        subjects = len(tr_data) #Params[Dataset[data]][0]
        tc =  1040   #Params[Dataset[data]][1]
        samples_per_subject = 52 # Params[Dataset[data]][2]
        window_shift = 20
        

        AllData = np.zeros((subjects, samples_per_subject, no_good_comp, sample_y))
        
        for i in range(subjects):
            for j in range(samples_per_subject):
                AllData[i, j, :, :] = finalData[i, :, (j * window_shift):(j * window_shift) + sample_y]

        print(AllData.shape)

        HC_index, SZ_index = find_indices_of_each_class(all_labels)
        #print('Data with overlapping windows: ',AllData.shape)
        #print('Length of HC: ', len(HC_index))
        #print('Length of SZ: ', len(SZ_index))


        # We will work here to change the split

        test_starts = { 'SYNDATA': [0, 32, 64, 96, 120], 'COBRE': [0, 16, 32, 48], 'OASIS': [0, 32, 64, 96, 120, 152], 'ABIDE': [0, 50, 100, 150, 200]}

        test_indices = test_starts[Dataset[data]]
        #print(f'Test Start Indices: {test_indices}')


        # for test_ID in range(1):

        test_ID = 4
        #test_ID = int(sys.argv[3])

        test_start_index = test_indices[test_ID]
        test_end_index = test_start_index + test_lim_per_class[Dataset[data]]
        total_HC_index_tr = torch.cat(
            [HC_index[:test_start_index], HC_index[test_end_index:]]
        )
        total_SZ_index_tr = torch.cat(
            [SZ_index[:test_start_index], SZ_index[test_end_index:]]
        )

        HC_index_test = HC_index[test_start_index:test_end_index]
        SZ_index_test = SZ_index[test_start_index:test_end_index]

        # total_HC_index_tr = HC_index[:len(HC_index) - test_lim_per_class[Dataset[data]]]
        # total_SZ_index_tr = SZ_index[:len(SZ_index) - test_lim_per_class[Dataset[data]]]

        #print('Length of training HC:', len(total_HC_index_tr))
        #print('Length of training SZ:', len(total_SZ_index_tr))

        # HC_index_test = HC_index[len(HC_index) - (test_lim_per_class[Dataset[data]]):]
        # SZ_index_test = SZ_index[len(SZ_index) - (test_lim_per_class[Dataset[data]]):]

                
        X = AllData
        Y = all_labels.numpy()

        #print('Data shape ended up with:', X.shape)

        #print("with LSTM enc + unidirectional lstm MILC + anchor point + sequence first...")




        np.random.seed(0)
        trials_HC = [np.random.permutation(len(total_HC_index_tr)) for i in range(Trials)]  
        trials_SZ = [np.random.permutation(len(total_SZ_index_tr)) for i in range(Trials)]  


        subjects_per_group = Params_subj_distr[Dataset[data]] 

        #print('SPC Info:', subjects_per_group)

        test_index = torch.cat((HC_index_test, SZ_index_test))
        #print(test_index)

        test_index = test_index.view(test_index.size(0))
        X_test = X[test_index, :, :, :]
        Y_test = Y[test_index.long()]
                                
        X_sal = X
        Y_sal = Y
                    
        X_test = torch.from_numpy(X_test).float().to(device)
        Y_test = torch.from_numpy(Y_test).long().to(device)

        X_sal = torch.from_numpy(X_sal).float().to(device)
        Y_sal = torch.from_numpy(Y_sal).long().to(device)

                    
        # X_full = torch.from_numpy(X).float().to(device)
        # Y_full = torch.from_numpy(Y).long().to(device)

        #print('Test Shape:', X_test.shape)
        #print(X_sal.shape)

        dataLoaderTest = get_data_loader(X_test, Y_test, X_test.shape[0])

        # ID = int(sys.argv[3])
        # g = GAIN[ID]

        accMat = np.zeros([len(subjects_per_group), Trials])
        aucMat = np.zeros([len(subjects_per_group), Trials])

        start_time = time.time()


        #print(f'Allocated: {torch.cuda.memory_allocated()}')

        #Best_gain = Params_best_gains[1][Dataset[data]][args.exp]
        print('Gain Values Chosen:', Gain)

        dir = args.exp   # NPT or UFPT
        wdb = 'wandb'
        wpath = os.path.join('../', wdb)
        sbpath = os.path.join(wpath, 'Sequence_Based_Models')

        #model_path = os.path.join(sbpath, Directories[Dataset[data]], dir)

        for i in range(len(Gain)):
            
            for restart in range(Trials):
                #print("Trial: ", restart)
                
                samples = subjects_per_group[i]
                
                g = Gain[i]
                print("Gain: ",g)
                

                HC_random = trials_HC[restart][:samples]  
                SZ_random = trials_SZ[restart][:samples]


                HC_index_tr = total_HC_index_tr[HC_random]
                SZ_index_tr = total_SZ_index_tr[SZ_random]




                tr_index = torch.cat((HC_index_tr, SZ_index_tr))
                

                #t = torch.squeeze(tr_index)
                #sorted, indices = torch.sort(t)
                #print("aa", len(sorted), sorted)

                #exit()

                tr_index = tr_index.view(tr_index.size(0))
                X_train = X[tr_index, :, :, :]
                Y_train = Y[tr_index.long()]

                X_train = torch.from_numpy(X_train).float().to(device)
                Y_train = torch.from_numpy(Y_train).long().to(device)

                #print('Train Data Shape:', X_train.shape)
                
                
                np.random.seed(0)
                randomize = np.random.permutation(X_train.shape[0])

                X_train_go = X_train[randomize]
                Y_train_go = Y_train[randomize]

                

                dataLoaderTrain = get_data_loader(X_train_go, Y_train_go, 32)
                #print('hehe', X_train_go.shape[0])
                dataLoaderTrainCheck = get_data_loader(X_train_go, Y_train_go, 32)
                
                #print(f'Test Split Starts: {test_ID}')

                #print('MILC + with TOP Anchor + both uniLSTM')
                #print(f'Model Started: {restart}\nSPC: {samples}\nGain: {g}\nExperiment MODE: {args.exp}\nDataset: {Dataset[data]}')

                ################################
                encoder = NatureOneCNN(53, args)
                lstm_model = subjLSTM(
                                            device,
                                            args.feature_size,
                                            args.lstm_size,
                                            num_layers=args.lstm_layers,
                                            freeze_embeddings=True,
                                            gain=g,
                                        ) 




                ###############################
                ################Model###################
                if encoderr == 'cnn':
                    
                    model = combinedModel(
                    encoder,
                    lstm_model,
                    gain=g,
                    PT=args.pre_training,
                    exp=args.exp,
                    device=device,
                    oldpath=args.oldpath,
                    complete_arc=args.complete_arc,
                )
                elif encoderr == 'lstmM':
                    #LSTM used by Mahfuz
                    model = LSTM(X.shape[2], 256, 200, 121, 2, g).float()
                ######################################################

                #PATH = '/data/users2/ziqbal5/MILC_LSTM/pretrainedModels/Trial_03687427_HCP_PTR.pt'
                #PATH = '/data/users2/ziqbal5/MILC_LSTM/pretrainedModels/Trial_03687430_HCP_PTR.pt'
                #PATH = '/data/users2/ziqbal5/MILC_LSTM/pretrainedModels/Trial_03687432_HCP_PTR.pt'
                optimizer, accMat[i, restart], aucMat[i, restart],model = train_model(model, dataLoaderTrain, dataLoaderTrainCheck, dataLoaderTest, eppochs, 3e-4)
                
                PATH = '/data/users2/ziqbal5/abc/MILC_LSTM/pretrainedModels/'
                #if args.exp == 'NPT':
                if encoderr == 'cnn':
                    #PATH = '/data/users2/ziqbal5/MILC_LSTM/pretrainedModels/Trial_03707479_HCP_PTR.pt'
                    torch.save(model.state_dict(),  os.path.join(PATH, str(args.jobid)+ '_S' + str(dd) + '_G' + str(i)+ '_CNN.pt'))  
                # if encoderr == 'lstm':
                #     #PATH = '/data/users2/ziqbal5/MILC_LSTM/pretrainedModels/LSTM_03716170_HCP_PTR.pt'
                #     torch.save(model.state_dict(),  os.path.join(PATH, str(args.jobid)+ '_'+ str(i)+'_' + str(restart)+ '_LSTM.pt'))  
                if encoderr == 'lstmM':
                    #PATH = '/data/users2/ziqbal5/MILC_LSTM/pretrainedModels/LSTMM_03716170_HCP_PTR.pt'
                    torch.save(model.state_dict(),  os.path.join(PATH, str(args.jobid)+ '_S' + str(dd) + '_G' + str(i)+ '_LSTMM.pt'))  

                    #print("Trained model loaded")
                    #model.load_state_dict(torch.load(PATH))
                # else:
                #     print("Model could not be saved")
                #print(f'Allocated: {torch.cuda.memory_allocated()}')
                


                middle_time = time.time() - start_time
                #print('Total Time for Training:', middle_time)
                
                prefix= f'{Dataset[data]}_spc_{subjects_per_group[i]}_simple_gain_{g}_window_shift_{window_shift}_{dir}_arch_chk_h_fixed_test_id_{test_ID}_LSTM_milc'

                if torch.cuda.is_available():
                    
                    #print(f'Allocated: {torch.cuda.memory_allocated()}')
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    #print(f'Allocated: {torch.cuda.memory_allocated()}')
                    
                
        basename2 = os.path.join('/data/users2/ziqbal5/abc/MILC_LSTM/', dir)

        #prefix = f'{Dataset[data]}_all_spc_cross_val_best_gains_window_shift_{window_shift}_{dir}_arch_chk_2_test_id_{test_ID}_LSTM_milc'
        prefix = f'{args.jobid}_{args.encoder}_{Dataset[data]}_window_shift_{window_shift}_{dir}_test_id_{test_ID}_LSTM_milc'

        # prefix = f'{Dataset[data]}_all_spc_all_gains_{GAIN[ID]}_window_shift_{window_shift}_{dir}_arch_chk_h_fixed_LSTM_milc'

        accDataFrame = pd.DataFrame(accMat)
        accfname = os.path.join(basename2, prefix +'_ACC.csv')
        accDataFrame.to_csv(accfname)
        print('Result Saved Here:', accfname)

        aucDataFrame = pd.DataFrame(aucMat)
        aucfname = os.path.join(basename2, prefix +'_AUC.csv')
        aucDataFrame.to_csv(aucfname)
            
        #print("AUC:", aucMat)
        #print("ACC:", accMat)

elapsed_time = time.time() - start_time
print('Total Time Elapsed:', elapsed_time)
print(datetime.now())