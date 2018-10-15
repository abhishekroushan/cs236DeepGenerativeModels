import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_lstm_units, num_lstm_layers, dataset, device):
        super().__init__()
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.h2o = nn.Linear(num_lstm_units, vocab_size)
        self.device = device
        self.dataset = dataset
        self.to(device)

    def forward(self, input, hidden=None):
        """
        Predict the next token's logits given an input token and a hidden state.
        :param input [torch.tensor]: The input token tensor with shape
            (batch_size, 1), where batch_size is the number of inputs to process
            in parallel.
        :param hidden [(torch.tensor, torch.tensor)]: The hidden state, or None if
            it's the first token.
        :return [(torch.tensor, (torch.tensor, torch.tensor))]: A tuple consisting of
            the logits for the next token, of shape (batch_size, num_tokens), and
            the next hidden state.
        """
        embeddings = self.embedding(input)
        if hidden is None:
            lstm, (h, c) = self.lstm(embeddings)
        else:
            lstm, (h, c) = self.lstm(embeddings, hidden)

        lstm = lstm.contiguous().view(-1, lstm.shape[2])
        logits = self.h2o(lstm)
        return logits, (h.detach(), c.detach())

    def sample(self, seq_len):
        """
        Sample a string of length `seq_len` from the model.
        :param seq_len [int]: String length
        :return [list]: A list of length `seq_len` that contains each token in order.
                        Tokens should be numbers from {0, 1, 2, ..., 656}.
        """
        voc_freq = self.dataset.voc_freq
        with torch.no_grad():
            # The starting hidden state of LSTM is None
            h_prev = None
            # Accumulate tokens into texts
            texts = []
            # Randomly draw the starting token and convert it to a torch.tensor
            x = np.random.choice(voc_freq.shape[0], 1, p=voc_freq)[None, :]
            x = torch.from_numpy(x).type(torch.int64).to(self.device)
            ##### Complete the code here #####
            # Append each generated token to texts
            # hint: you can use self.forward
            texts.append(int(x))
            char_pred_idx=x
            h=h_prev
            for i in range(1,seq_len):
                logits,h=self.forward(char_pred_idx, h)
                probs=F.softmax(logits, dim=1)#axis
                #probs_ary=probs.reshape(-1)
                #Rounding error to proobabilities sum (float), therefore normalizing
                probs_ary=probs.numpy()
                probs_ary /=probs_ary.sum(axis=1)
                probs_ary=probs_ary.reshape(-1)
                char_pred_idx=np.random.choice(voc_freq.shape[0],1,p=probs_ary)[None,:]
                char_pred_idx=torch.from_numpy(char_pred_idx).type(torch.int64).to(self.device)
                texts.append(int(char_pred_idx))
            ##################################

        return texts

    def compute_prob(self, string):
        """
        Compute the probability for each string in `strings`
        :param string [np.ndarray]: an integer array of length N.
        :return [float]: the log-likelihood
        """
        voc_freq = self.dataset.voc_freq
        with torch.no_grad():
            # The starting hidden state of LSTM is None
            h_prev = None
            # Convert the starting token to a torch.tensor
            x = string[None, 0, None]
            x = torch.from_numpy(x).type(torch.int64).to(self.device)
            # The log-likelihood of the first token.
            # You should accumulate log-likelihoods of all other tokens to ll as well.
            ll = np.log(voc_freq[string[0]])
            ##### Complete the code here ######
            # Add the log-likelihood of each token into ll
            # hint: you can use self.forward
            #ll = torch.FloatTensor([ll])
            h=h_prev
            for i in string:
                logits,h=self.forward(x,h)
                probs=F.softmax(logits, dim=1)
                probs=probs.reshape(-1)
                probs=probs.numpy()
                x = np.array(i).reshape(1,1)
                x = torch.from_numpy(x).type(torch.int64).to(self.device)


                ll+=np.log(probs[i])
                #Another implementation:
                #logProbs=F.log_softmax(logits, dim=1)
                #logProbs=logProbs.reshape(-1)
                #ll+=logProbs[i]
            ###################################

            return ll
