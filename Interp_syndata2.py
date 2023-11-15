from ast import Load
from captum.attr import IntegratedGradients, GradientShap, NoiseTunnel, Occlusion
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
from matplotlib.backends.backend_pdf import PdfPages
import torchvision.transforms as T
import sys
sys.path.append("..")
from PIL import Image
import pickle
from pathlib import Path
import numpy as np
import torch
import mat73
from wholeMILC import NatureOneCNN
from lstm_attn import subjLSTM
from utils import get_argparser
from All_Architecture import combinedModel
from torch.utils.data import DataLoader, TensorDataset
import os
from tqdm import tqdm
import pandas as pd
#from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.patches as patches
import warnings
from enum import Enum
from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
from matplotlib import cm, colors, pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray

try:
    from IPython.display import display, HTML

    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False
    

parser = get_argparser()
args = parser.parse_args()
print("JOBID: ", args.jobid)
#Path to store extracted attributes
if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
else:
        device = torch.device("cpu")
class ImageVisualizationMethod(Enum):
    heat_map = 1
    blended_heat_map = 2
    original_image = 3
    masked_image = 4
    alpha_scaling = 5



class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


def _prepare_image(attr_visual: ndarray):
    return np.clip(attr_visual.astype(int), 0, 255)


def _normalize_scale(attr: ndarray, scale_factor: float):
    #print("ab: ", attr.shape)
    #print(scale_factor)
    assert scale_factor != 0, "Cannot normalize by scale factor = 0"
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, visualized results"
            "may be misleading. This likely means that attribution values are all"
            "close to 0."
        )
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)
    


def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def _normalize_attr(
    attr: ndarray,
    sign: str,
    outlier_perc: Union[int, float] = 2,
    reduction_axis: Optional[int] = None,
):
    attr_combined = attr
    #print('a', attr_combined.shape)
    #print("rax", reduction_axis)
    if reduction_axis is not None:
        attr_combined = np.sum(attr, axis=reduction_axis)
    #print('a', attr_combined.shape)
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)


def visualize_image_attr(
    attr: ndarray,
    original_image: Union[None, ndarray] = None,
    method: str = "heat_map",
    sign: str = "absolute_value",
    plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
    outlier_perc: Union[int, float] = 2,
    cmap: Union[None, str] = None,
    alpha_overlay: float = 0.5,
    show_colorbar: bool = False,
    title: Union[None, str] = None,
    fig_size: Tuple[int, int] = (6, 6),
    use_pyplot: bool = True,
):
 
    heat_map = None

    norm_attr = _normalize_attr(attr, sign, outlier_perc, reduction_axis=2)



    return norm_attr, norm_attr, norm_attr




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
        
        #print('Initializing All components')
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
            #print('Initializing Decoder:', name)
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


def Load_Data():
    with open(os.path.join(path_a, str(jobidd)+ '_' + str(dd)+'test_data.pickle'), "rb") as infile:
        X = pickle.load(infile)
    with open(os.path.join(path_a, str(jobidd)+ '_' + str(dd)+'test_labels.pickle'), "rb") as infile:
        L = pickle.load(infile)
        
    no_good_comp = 53
    sample_y = 20
    subjects = len(X) #Params[Dataset[data]][0]
    tc =  140   #Params[Dataset[data]][1]
    samples_per_subject = 121 # Params[Dataset[data]][2]
    window_shift = 1

    AllData = np.zeros((subjects, samples_per_subject, no_good_comp, sample_y))

    for i in range(subjects):
        for j in range(samples_per_subject):
            AllData[i, j, :, :] = X[i, :, (j * window_shift):(j * window_shift) + sample_y]
        C_data = torch.from_numpy(AllData).float().to(device)
        #L = torch.from_numpy(L).long().to(device)
        #print(C_data.shape, L.shape, type(C_data), type(L))
    return C_data, L
   

def Initiate_Model(path_models, gain, encoderr):
    #Initiate Model
    sample_x = 53
    current_gain = gain
    
    encoder = NatureOneCNN(sample_x, args)
    lstm_model = subjLSTM(
                        device,
                        args.feature_size,
                        args.lstm_size,
                        num_layers=args.lstm_layers,
                        freeze_embeddings=True,
                        gain=current_gain,
                    )
    if encoderr == 'cnn':    
        model = combinedModel(
            encoder,
            lstm_model,
            gain=current_gain,
            PT=args.pre_training,
            exp=args.exp,
            device=device,
            oldpath=args.oldpath,
            complete_arc=args.complete_arc,
        )
    
    elif encoderr == 'lstmM':
            #LSTM used by Mahfuz
            model = LSTM(53, 256, 200, 121, 2, current_gain).float()
 
    path_m = path_models
    #print(path_m)
    model.load_state_dict(torch.load(path_m))
    print("Model trained on synthetic data loaded from: ", path_m)

    return model

def Predicted_Labels(model, TestDataFD, LabelsFD, pathh):
        #Load Model pretrained weights

 

    #print("aadfasfsa", TestDataFD.shape, LabelsFD.shape)
    datasetFD = TensorDataset(TestDataFD, LabelsFD)
    loaderSalFD = DataLoader(datasetFD, batch_size=subjects, shuffle=False)


    model.eval()
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for i, data in enumerate(loaderSalFD):
        x, y = data
        x=torch.squeeze(x)
        
        x = x.to(device)
        y = y.to(device)
       
        
        
        outputs = model(x)
        _, predsFD = torch.max(outputs.data, 1)
        #print(predsFD)
    with open(os.path.join(pathh, 'Sal_IG', str(jobidd) + 'pred_labels_S' + str(dd) + '_G' + str(g)+ '.pickle'), "wb") as outfile:
        pickle.dump(predsFD, outfile)
    return predsFD





def Feature_Attributions(model, TestDataFD, LabelsFD, predsFD):


    datasetFD = TensorDataset(TestDataFD, predsFD)
    loaderSalFD = DataLoader(datasetFD, batch_size=1, shuffle=False)

    
    model.eval()
    model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
   
    salienciesFD = []
    
    for i, data in enumerate(loaderSalFD):
        if i % 10 == 0:
            print(f'Processing subject:{i}')
        x, y = data 
        
        #x=torch.squeeze(x, dim=0)
        
        x = x.to(device)
        y = y.to(device)
            
        #print("aa",x.shape, predsFD.shape)
        #predsFD= torch.unsqueeze(predsFD, dim=0)
        #print(predsFD)
        bl = torch.zeros(x.shape).to(device)
                
        
        x = x.to(device)
        y = y.to(device)
        
        model.train()
        x.requires_grad_()

        # Integrated Gradeints
        sal = IntegratedGradients(model)
        attribution = sal.attribute(x,bl,target=y)

        #GradientShap
        # sal = GradientShap(model)
        # attribution = sal.attribute(x,bl, stdevs = 0.09,target=y)

        #NoiseTunnel
        #ig = IntegratedGradients(model)
        #sal = NoiseTunnel(ig)
        #attribution = sal.attribute(x,nt_samples=10, nt_type ="smoothgrad_sq", target=y)

   


        salienciesFD.append(np.squeeze(attribution.cpu().detach().numpy()))

        
        
    all_salienciesFD = np.stack(salienciesFD, axis=0)
    #print(all_salienciesFD.shape)
    
    with open(os.path.join(pathh, 'Sal_IG',  str(jobidd) + 'all_saliencies_S' + str(dd) + '_G' + str(g) + '.pickle'), "wb") as outfile:
        pickle.dump(all_salienciesFD, outfile)

def stitch_windows(saliency, components, samples_per_subject, sample_y, time_points, ws):

    stiched_saliency = np.zeros((saliency.shape[0], components, samples_per_subject * sample_y))
    for i in range(saliency.shape[0]):
        for j in range(saliency.shape[1]):
            stiched_saliency[i, :, j * 20:j * 20 + sample_y] = saliency[i, j, :, :]

    saliency = stiched_saliency

    avg_saliency = np.zeros((saliency.shape[0], components, time_points))

    if ws == 20:
        avg_saliency = saliency

    elif ws == 10:
        avg_saliency[:, :, 0:10] = saliency[:, :, 0:10]

        for j in range(samples_per_subject-1):
            a = saliency[:, :, 20*j+10:20*j+20]
            b = saliency[:, :, 20*(j+1):20*(j+1)+10]
            avg_saliency[:, :, 10*j+10:10*j+20] = (a + b)/2

        avg_saliency[:, :, time_points-10:time_points] = saliency[:, :, samples_per_subject*sample_y-10:samples_per_subject*sample_y]

    else:
        for i in range(saliency.shape[0]):
            for j in range(time_points):
                L = []
                if j < 20:
                    index = j
                else:
                    index = 19 + (j - 19) * 20

                L.append(index)

                s = saliency[i, :, index]
                count = 1
                block = 1
                iteration = min(19, j)
                for k in range(0, iteration, 1):
                    if index + block * 20 - (k + 1) < samples_per_subject * sample_y:
                        s = s + saliency[i, :, index + block * 20 - (k + 1)]
                        L.append(index + block * 20 - (k + 1))
                        count = count + 1
                        block = block + 1
                    else:
                        break
                avg_saliency[i, :, j] = s/count
                # print('Count =', count, ' and Indices =', L)

    return avg_saliency


def visualization(path_sm, path_labels, ik, encoding):
    # img_ind = 3
    #print("visualization")
    #sal = [1,2,3,4,5,6,7] #dummy values to set the size of the list.
    sal = list(range(seeds))
    pred = list(range(seeds))
    #pred = [1,2,3,4,5,6,7]
    components, samples_per_subject, sample_y, time_points, ws = 53,121,20,140,1
    for i in range(len(path_sm)):
        with open(path_sm[i], "rb") as infile:
            sal[i] = pickle.load(infile)
            #print("shape of saliency maps before stiching: ", sal[i].shape)
            sal[i] = stitch_windows(sal[i], components, samples_per_subject, sample_y, time_points, ws)
            
        with open(path_labels[i], "rb") as infile:
            pred[i] = pickle.load(infile)


    #subjectss, window_num, comp, window_time  = subjects, 7, 53, 20
    test_data = list(range(seeds))
    pos =       list(range(seeds))
    pathtotestdata = '/data/users2/ziqbal5/abc/MILC_LSTM/Data/'
    #3824232_0test_data.pickle'
    for i in range(seeds):
        with open(os.path.join(pathtotestdata, str(jobidd)+'_'+str(i)+'test_data.pickle'), "rb") as infile:
            test_data[i] = pickle.load(infile)
        with open(os.path.join(pathtotestdata, str(jobidd)+'_'+str(i)+'s_position.pickle'), "rb") as infile:
            pos[i] = pickle.load(infile)
            pos[i] = pos[i][2000:2500]

    for i in range(seeds):
        # #Normalize values
        all_sal = []
        for index, data in enumerate(sal[i]):
            #print('index', index)
            data = torch.Tensor(data)
            #print(index, data.shape)
            outmap_min = torch.min(data)
            outmap_max = torch.max(data)
            #Normalize [0, 1]
            # sal_n = ((data - outmap_min) / (outmap_max - outmap_min))
            # sal[i][index] = sal_n
            #Normalize [-1, 1]
            # sal_n = 2 * ((data - outmap_min) / (outmap_max - outmap_min)) - 1
            # sal[i][index] = sal_n
            # mean = data.mean
            # sd = data.std()
            # sal_n = (data - mean)/sd
            # sal[i][index] = sal_n
        
    
    onepager = 10 #Number of rows in one page
    fig, axs = plt.subplots(onepager, 2*len(sal),  facecolor='w', edgecolor='k', figsize=(15,15))
    fig.subplots_adjust(hspace = .5, wspace=.001)
    fig.suptitle(["Encoder: ", encoding, "Gain: ", Gain2[counter2]], fontsize=16 )
    #a = 0    
    
    #cols = ['seed1', 'seed2', 'seed3', 'seed4', 'seed5', 'seed6', 'seed7']
    
    countt = 0
    cmap = 'bwr'
    viz_img_attr = False
    for i in range(onepager):
        #for j in range(len(cols)):
        for j in range(2*seeds):
            if viz_img_attr:
                sall = sal[countt][i]
                sall = np.expand_dims(sall, axis = 2)
                imgg = test_data[countt][i]
                imgg = imgg.numpy()
                imgg = np.expand_dims(imgg, axis = 2)
                #print(sall.shape, imgg.shape, type(sall), type(imgg))

                ab = visualize_image_attr(sall, imgg, cmap = cmap, method="heat_map",sign="absolute_value",
                                    show_colorbar=True, title="Overlayed Integrated Gradients")
                sal[countt][i] = ab[2]

            if pred[countt][i] == 0:
                m = 30
            else:
                m = 0
            rect = patches.Rectangle((pos[countt][i], m), 10, 23, linewidth=1, edgecolor='r', facecolor='none')
            if j %2 == 0:
                abb = axs[i,j].imshow(test_data[countt][i], cmap = cmap)
                #if countt == 0:
                    #ac = plt.colorbar(abb, ax=axs[i,j], location = 'top')
                    #ac.set_ticks([])
                axs[i,j].add_patch(rect)
                if i == 0:
                    axs[i,j].set_title('Data')

            else:
                ######################
                # # #sal[countt][i] = _normalize_attr(sal[countt][i], "absolute_value", 2, reduction_axis=None)
                # attr_combined= np.expand_dims(sal[countt][i], axis = 2)
                # #print(attr_combined.shape, type(attr_combined))
                # attr_combined = np.sum(attr_combined, axis=2)
                # #print(attr_combined.shape, type(attr_combined))
                # attr_combined = np.abs(attr_combined)
                # threshold = _cumulative_sum_threshold(attr_combined, 100 - 2)
                # sal[countt][i] = _normalize_scale(attr_combined, threshold)
                
                #################
                if viz_img_attr == False:
                    sal[countt][i] = np.abs(sal[countt][i])  #take abolute values\
                #sal[countt][i] = (sal[countt][i] > 0) * sal[countt][i] #take positive values
                #sal[countt][i] = (sal[countt][i] < 0) * sal[countt][i] #take negative values
                axs[i,j].imshow(sal[countt][i], cmap = cmap)
                axs[i,j].add_patch(rect)
                if i == 0:
                    axs[i,j].set_title('Sal Maps')
                countt = countt+1
            #axs[i,j].set_title(int(predLL[i]), loc='left')
            #axs[i,j].set_title([int(trueLL[i]), int(predLL[i])])
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])
        countt = 0
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7]) #(left, bottom, width, height) 
    cbar = fig.colorbar(abb, cax=cbar_ax)
    cbar.set_ticks([])

        
    pathtt = '/data/users2/ziqbal5/abc/MILC_LSTM/Sal_CNN/Figures_IG/'
    #plt.savefig(pathtt)
    plt.savefig(pathtt + str(jobidd) + 'sal_'+str(ik)+'.png')
    #a = a+onepager
    
    plt.close()
    
    
    
if __name__ == "__main__":

    seeds = 3
    enc = ['lstmM', 'cnn']
    Gain = [1.2, 0.9, 0.7, 0.65] # This is for jobid : 3915997
    #Gain = [0.9, 0.7, 0.6, 0.5, 0.4, 0.05] # This is for jobid : 3824249
    subjects = 50  #taking 50 test subjects to calculate saliency maps
    #jobidd = str(3824249) 
    jobidd = str(3915997)
    Trials = 1

    if args.viz == False:
        for dd in range(seeds):
            print("Seed: ", dd)
            for en in range(len(enc)):
                encoderr = enc[en]
                print("Encoder: ", encoderr)

                if encoderr == 'lstmM':
                    path_a = Path('/data/users2/ziqbal5/abc/MILC_LSTM/Data/')
                    pathh = '/data/users2/ziqbal5/abc/MILC_LSTM/Sal_LSTMM'
                    
                    name = '_LSTMM'
                elif encoderr == 'cnn':
                    path_a = Path('/data/users2/ziqbal5/abc/MILC_LSTM/Data/')
                    pathh = '/data/users2/ziqbal5/abc/MILC_LSTM/Sal_CNN'
                    name = '_CNN'
                
            
                TestDataFD, LabelsFD = Load_Data()
                TestDataFD = TestDataFD[0:50]
                LabelsFD = LabelsFD[0:50]
                #print(TestDataFD.shape, type(TestDataFD), LabelsFD.shape)

        
                PATH = '/data/users2/ziqbal5/abc/MILC_LSTM/pretrainedModels/'
                # if args.viz:

                #     for j in range(Trials):
                #         for g in range(len(Gain)):
                #             path_sm = os.path.join(pathh, 'Sal_IG', 'all_saliencies_S' + str(dd)+ '_G' + str(g)+ '.pickle')
                #             path_labels = os.path.join(pathh, 'sal_IG', 'pred_labels_S' + str(dd)+ '_G' + str(g)+ '.pickle')
                #             visualization(path_sm, path_labels)
            
                for j in range(Trials):
                    for g in range(len(Gain)):
                    
                        path_models = os.path.join(PATH, jobidd + '_S' + str(dd)+ '_G' + str(g) + name +'.pt')
                        gain = Gain[g]
                        model = Initiate_Model(path_models, gain, encoderr) 
                        predsFD = Predicted_Labels(model, TestDataFD, LabelsFD, pathh)
                        Feature_Attributions(model, TestDataFD,LabelsFD, predsFD)
                        

    if args.viz:
        Gain2 =  [val for val in Gain for _ in (0, 1)] #Doubling the values in th list.
        all_sal_paths = []
        all_testdata_paths = []
        for g in range(len(Gain)):    
            for en in range(len(enc)):
                for dd in range(seeds):
                #for g in range(len(Gain)):
                    encoderr = enc[en]
                    if encoderr == 'cnn':
                        pathtosal = '/data/users2/ziqbal5/abc/MILC_LSTM/Sal_CNN/'
                    else:
                        pathtosal = '/data/users2/ziqbal5/abc/MILC_LSTM/Sal_LSTMM/'
                #for g in range(len(Gain)):
                    path_sm = os.path.join(pathtosal,     'Sal_IG', str(jobidd) +'all_saliencies_S' + str(dd)+ '_G' + str(g)+ '.pickle')
                    all_sal_paths.append(path_sm) 
                    path_labels = os.path.join(pathtosal, 'Sal_IG', str(jobidd) + 'pred_labels_S' + str(dd)+ '_G' + str(g)+ '.pickle')
                    all_testdata_paths.append(path_labels)

        counter = 0  #No of columns in one page (seeds)
        counter2 = 0  #No of pages(experiments)
        pathtosal = []
        pathtopred = []
        #for i in range(16):
        for i in range(len(all_sal_paths)):
            
            if 'Sal_CNN' in all_sal_paths[i]:
                encoding = 'CNN'
                pathtosal.append(all_sal_paths[i])
                pathtopred.append(all_testdata_paths[i])
            else:
                encoding = 'LSTM'
                pathtosal.append(all_sal_paths[i])
                pathtopred.append(all_testdata_paths[i])
            counter = counter+1
            if counter == seeds:
                #print(pathtosal)
                #print("ab")
                counter = 0
                print("Page: ", counter2)
                print ('\n'.join(pathtosal)) 
                #print('\n\n')
                #print ('\n'.join(pathtosal)) 
                
                visualization(pathtosal, pathtopred, counter2, encoding)
                counter2 = counter2+1
                pathtosal = []
                pathtopred = []
            #if i == 6:
                #pass
                #exit()
    
                
        
            


