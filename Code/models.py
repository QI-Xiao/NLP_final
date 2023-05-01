import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertModel


checkpoint = "distilbert-base-uncased"


class BertRNNModel(nn.Module):
    def __init__(self, checkpoint, num_labels):
        super(BertRNNModel, self).__init__()
        self.bert = BertModel.from_pretrained(checkpoint)
        self.lstm = nn.LSTM(768, 256, 2,
                            bidirectional=True, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc_rnn = nn.Linear(256 * 2, num_labels)

    def forward(self, **batch):
        # encoder_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        encoder_out = self.bert(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        out, _ = self.lstm(encoder_out[1])
        out = self.dropout(out)
        out = self.fc_rnn(out)  # hidden state
        return out
