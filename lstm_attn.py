import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.utils.rnn as tn


class subjLSTM(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(
        self,
        device,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        freeze_embeddings=True,
        gain=1,
    ):

        super(subjLSTM, self).__init__()
        self.gain = gain
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.freeze_embeddings = freeze_embeddings

        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.hidden_dim // 2,
            num_layers=num_layers,
            bidirectional=True,
        )

        # The linear layer that maps from hidden state space to tag space
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.hidden_dim, 200),
        #     nn.ReLU(),
        #     nn.Linear(200, 2)
        #
        # )
        self.init_weight()

    def init_hidden(self, batch_size, device):
        h0 = Variable(
            torch.zeros(2, batch_size, self.hidden_dim // 2, device=device)
        )
        c0 = Variable(
            torch.zeros(2, batch_size, self.hidden_dim // 2, device=device)
        )
        return (h0, c0)

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=self.gain)
        # for name, param in self.decoder.named_parameters():
        #     if 'weight' in name:
        #         nn.init.xavier_normal_(param, gain=self.gain)

    def forward(self, inputs, mode="train"):
        #print("lstminputsshape", inputs[0].shape, len(inputs)) #7x256, 30 
        packed = tn.pack_sequence(inputs, enforce_sorted=False)

        self.hidden = self.init_hidden(len(inputs), packed.data.device)
        self.lstm.flatten_parameters()
        if mode == "eval" or mode == "test":
            with torch.no_grad():
                packed_out, self.hidden = self.lstm(packed, self.hidden)
        else:
            packed_out, self.hidden = self.lstm(packed, self.hidden)

        # output, lens = tn.pad_packed_sequence(packed_out, batch_first=True, total_length=total_length)
        outputs, lens = tn.pad_packed_sequence(packed_out, batch_first=True)
        # outputs = [line[:l] for line, l in zip(outputs, lens)]
        # outputs = [self.decoder(torch.cat((x[0, self.hidden_dim // 2:],
        #                                    x[-1, :self.hidden_dim // 2]), 0)) for x in outputs]
        #print("lstmoutputs", outputs.shape) #30x7x200
        return outputs
