import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lstm_attn import subjLSTM
from All_Architecture import combinedModel
import pickle
from pathlib import Path
import os

sample_x = 53
from utils import get_argparser
parser = get_argparser()

args = parser.parse_args()
tags = ["pretraining-only"]
config = {}
config.update(vars(args))

if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        device = torch.device("cuda:" + cudaID)
        
else:
        device = torch.device("cpu")

print('device = ', device)

if args.exp == 'FPT':
    gain = [0.1, 0.05, 0.05]  # FPT
elif args.exp == 'UFPT':
    gain = [0.05, 0.45, 0.65]  # UFPT
else:
    gain = [0.25, 0.35, 0.65]  # NPT
ID = args.script_ID - 1
current_gain = gain[ID]
args.gain = current_gain

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class NatureOneCNN(nn.Module):
    def init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

    def init_hidden_enc(self, batch_size, device):
        
                h0 = Variable(torch.zeros(1, batch_size, 256, device=device))
                c0 = Variable(torch.zeros(1, batch_size, 256, device=device))
        
                return (h0, c0)

    def get_attention_enc(self, outputs):
        
        # For anchor point
        B= outputs[:,-1, :]
        B = B.unsqueeze(1).expand_as(outputs)
        outputs2 = torch.cat((outputs, B), dim=2)
        
        
        # For attention calculation
        b_size = outputs.size(0)
        # out = outputs.view([-1, self.hidden])
        out = outputs2.reshape(-1, 2* self.enc_out)

        weights = self.attnenc(out)
        weights = weights.view(b_size, -1, 1)
        weights = weights.squeeze(2)

        # SoftMax normalization on weights
        normalized_weights = F.softmax(weights, dim=1)

        # Batch-wise multiplication of weights and lstm outputs

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.squeeze()

        return attn_applied

    def __init__(self, input_channels, args):
        super().__init__()
        self.feature_size = args.feature_size
        #print("__________", self.feature_size)
        self.input_size = 53
        self.enc_out = 256
        self.encoderr = args.encoder    #use cnn or lstm keywords
        self.hidden_size = self.feature_size
        self.downsample = not args.no_downsample
        self.input_channels = input_channels
        self.twoD = args.fMRI_twoD
        self.end_with_relu = args.end_with_relu
        self.args = args
        init_ = lambda m: self.init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )
        self.flatten = Flatten()

        if self.downsample:
            self.final_conv_size = 32 * 7 * 7
            self.final_conv_shape = (32, 7, 7)
            self.main = nn.Sequential(
                init_(nn.Conv2d(input_channels, 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, stride=1)),
                nn.ReLU(),
                Flatten(),
                init_(nn.Linear(self.final_conv_size, self.feature_size)),
                # nn.ReLU()
            )

        else:
            #print("fet size", self.feature_size)
            self.final_conv_size =  200*12 #26400
            self.final_conv_shape = (200, 12)
            #CNN Encoder start
            if self.encoderr == "cnn":  

                self.main = nn.Sequential(
                    init_(nn.Conv1d(input_channels, 64, 4, stride=1)),
                    nn.ReLU(),
                    init_(nn.Conv1d(64, 128, 4, stride=1)),
                    nn.ReLU(),
                    init_(nn.Conv1d(128, 200, 3, stride=1)),
                    nn.ReLU(),
                    Flatten(),
                    
                    init_(nn.Linear(self.final_conv_size, self.feature_size)),
                    init_(nn.Conv1d(200, 128, 3, stride=1)),
                    nn.ReLU(),
                    # nn.ReLU()
                )
            # #CNN Encoder End
            else:
            #LSTM encoder start
                self.encoder = nn.LSTM(self.input_size, self.enc_out, batch_first = True)
                self. attnenc = nn.Sequential(
                nn.Linear(2*self.enc_out, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
                )
            #LSTM encoder End
            
        #     from torch.autograd import Variable

        #     if torch.cuda.is_available():
        #             cudaID = str(torch.cuda.current_device())
        #             device = torch.device("cuda:" + cudaID)
        #     else:
        #             device = torch.device("cpu")            
        #     def init_hidden_enc(batch_size, device):
        
        #         h0 = Variable(torch.zeros(1, batch_size, 256, device=device))
        #         c0 = Variable(torch.zeros(1, batch_size, 256, device=device))
        
        #         return (h0, c0)
        # self.enc_hidden = init_hidden_enc(30, device)
        # #print(self.enc_hidden[0].shape, self.enc_hidden[1].shape) #1x30x256 both
        # self.encoder = nn.LSTM(52, 256, batch_first = True)  
        # out, self.enc_hidden = self.encoder(x, self.enc_hidden)
        
        

            

        # self.train()

    def forward(self, inputs, fmaps=False, five=False):
        #print('input: ', type(inputs), inputs.shape)
        if self.encoderr == 'lstm':
        #####For LSTM Encoder Start########## 
            input_batch = inputs
            input_batch = input_batch.permute(0, 2, 1)
            enc_batch_size = input_batch.size(0)
            if torch.cuda.is_available():
                        cudaID = str(torch.cuda.current_device())
                        device = torch.device("cuda:" + cudaID)
            else:
                        device = torch.device("cpu")
            enc_hidden = self.init_hidden_enc(enc_batch_size, device)
            out, enc_hidden = self.encoder(input_batch, enc_hidden)
            out = self.get_attention_enc(out)
            #print(out.shape)
         ######For LSTM Encoder End########## 

    
        
        else:    
    ######For CNN Encoder Start##########    
            f5 = self.main[:6](inputs)
            
            out = f5
            #print("f5",f5.shape)
            out = self.main[6:8](f5)
            #print(out.shape)
            f5 = self.main[8:](f5)

            if self.end_with_relu:
                assert (
                    self.args.method != "vae"
                ), "can't end with relu and use vae!"
                out = F.relu(out)
            if five:
                return f5.permute(0, 2, 1)
            if fmaps:
                return {
                    "f5": f5.permute(0, 2, 1),
                    # 'f7': f7.permute(0, 2, 1),
                    "out": out,
                }
            #print("out", out.shape)    #52(no of windows) x 256
    #####For CNN Encoder End##########
        return out


# encoder = NatureOneCNN(sample_x, args)
# #print(encoder)

# lstm_model = subjLSTM(
#                             device,
#                             args.feature_size,
#                             args.lstm_size,
#                             num_layers=args.lstm_layers,
#                             freeze_embeddings=True,
#                             gain=current_gain,
#                         )

# #print(lstm_model)

# complete_model = combinedModel(
#     encoder,
#     lstm_model,
#     gain=current_gain,
#     PT=args.pre_training,
#     exp=args.exp,
#     device=device,
#     oldpath=args.oldpath,
#     complete_arc=args.complete_arc,
# )
# path_a = Path('/data/users2/ziqbal5/MILC_LSTM/Models/')
# with open(os.path.join(path_a, 'LSTM_Model.pickle'), "wb") as outfile:
#     pickle.dump(complete_model, outfile)

# with open(os.path.join(path_a, 'LSTM_Model.pickle'), "rb") as infile:
#             Alll = pickle.load(infile)
