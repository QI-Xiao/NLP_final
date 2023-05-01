import os
import collections

import torch
from torch.utils import data
import torch.nn.functional as F

from transformers import BertForSequenceClassification, DistilBertForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

from tqdm.auto import tqdm
from datasets import load_metric

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dic_labels = {'anger': 0, 'love': 1, 'joy': 2, 'surprise': 3, 'fear': 4, 'sadness': 5}

os.chdir("..")
root_path = os.getcwd() + os.path.sep + 'Data' + os.path.sep


def get_data(data_type):
    path = root_path + data_type + '.txt'

    with open(path, 'r') as f:
        data = f.readlines()

    sentences = []
    labels = []

    for item in data:
        s, l = item.strip().split(';')
        sentences.append(s)
        labels.append(dic_labels[l])

    print(path, collections.Counter(labels))

    inputs = tokenizer(sentences, truncation=True)  # , padding='max_length'

    data_set = Dataset(inputs, data_type, labels)

    shuffle = True if data_type == 'train' else False
    data_loader = torch.utils.data.DataLoader(data_set, shuffle=shuffle, batch_size=32, collate_fn=data_collator)
    return data_loader, labels

class Dataset(data.Dataset):
    '''
    From : https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
    '''
    def __init__(self, encodings, type_data, labels):
        self.encodings = encodings
        self.type_data = type_data  # train test val
        self.labels = labels

    def __len__(self):
        #Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        #Generates one sample of data'
        # Select sample
        # sentence = self.sentence_list[index]
        # label = self.target_type[index]

        # return sentence, label
        item = {k: torch.tensor(v[index]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index])
        return item


train_dataloader, train_labels = get_data('train')
test_dataloader, test_labels = get_data('test')
val_dataloader, val_labels = get_data('val')

# for batch in train_dataloader:
#     break
#
# # print(batch)
# print({k: v.shape for k, v in batch.items()})

# model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=6)

from models import BertRNNModel
model = BertRNNModel('bert-base-uncased', 6)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# outputs = model(**batch)
# print(outputs.loss, outputs.logits.shape)


num_epochs = 3

num_training_steps = num_epochs * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)


model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        if True:
            loss = F.cross_entropy(outputs, batch['labels'])
        else:
            loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

metric = load_metric("glue", "mrpc")
model.eval()

pred_lst = []
for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    if True:
        logits = outputs
    else:
        logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    pred_lst.extend(predictions.tolist())

res = f1_score(val_labels, pred_lst, average='micro')
print(res)
# print(metric.compute(average='micro'))