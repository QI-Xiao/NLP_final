import os
import collections

import torch
from torch.utils import data
import torch.nn.functional as F

import transformers
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef

from tqdm.auto import tqdm
from datasets import load_metric

import shap

from models import BertRNNModel

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
    return data_loader, labels, sentences

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


train_dataloader, train_labels, train_sentences = get_data('train')
test_dataloader, test_labels, _ = get_data('test')
val_dataloader, val_labels, _ = get_data('val')


model_bert_lstm = False

if model_bert_lstm:
    # model bert with lstm
    model = BertRNNModel('bert-base-uncased', 6)
    print('using model bert with lstm')
else:
    # model bert classification
    checkpoint = "distilbert-base-uncased"
    model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=6)
    print('using model bert classification')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)


num_epochs = 5

num_training_steps = num_epochs * len(train_dataloader)

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)

metric = load_metric("glue", "mrpc")

train_loss_lst = []
test_loss_lst = []
acc_train_lst = []
acc_test_lst = []



for epoch in range(num_epochs):

    train_loss, steps_train = 0, 0
    pred_train_lst = []
    origin_train_lst = []

    model.train()
    with tqdm(total=len(train_dataloader), desc="Training Epoch {}".format(epoch)) as pbar:
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if model_bert_lstm:
                loss = F.cross_entropy(outputs, batch['labels'])
            else:
                loss = outputs.loss
                outputs = outputs.logits

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)

            train_loss += loss
            steps_train += 1

            predictions = torch.argmax(outputs, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            pred_train_lst.extend(predictions.tolist())
            origin_train_lst.extend(batch['labels'].tolist())

    avg_train_loss = (train_loss / steps_train).item()
    acc_train = f1_score(pred_train_lst, origin_train_lst, average='micro')


    model.eval()
    pred_test_lst = []
    test_loss, steps_test = 0, 0

    with tqdm(total=len(test_dataloader), desc="Test Epoch {}".format(epoch)) as pbar:
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

                if model_bert_lstm:
                    loss = F.cross_entropy(outputs, batch['labels'])
                else:
                    loss = outputs.loss
                    outputs = outputs.logits

            pbar.update(1)

            test_loss += loss
            steps_test += 1

            predictions = torch.argmax(outputs, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            pred_test_lst.extend(predictions.tolist())

    acc_test = f1_score(pred_test_lst, test_labels, average='micro')
    avg_test_loss = (test_loss / steps_test).item()

    print('epoch' + str(epoch), avg_train_loss, acc_train, avg_test_loss, acc_test, '\n')

    train_loss_lst.append(avg_train_loss)
    test_loss_lst.append(avg_test_loss)
    acc_train_lst.append(acc_train)
    acc_test_lst.append(acc_test)


print('model evaluation')
pred_val_lst = []
for batch in val_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

        if model_bert_lstm:
            loss = F.cross_entropy(outputs, batch['labels'])
        else:
            loss = outputs.loss
            outputs = outputs.logits

    predictions = torch.argmax(outputs, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])
    pred_val_lst.extend(predictions.tolist())

acc_val = f1_score(pred_val_lst, val_labels, average='micro')
acc_val2 = f1_score(pred_val_lst, val_labels, average='macro')
print('val accuracy:', acc_val, acc_val2)


import matplotlib.pyplot as plt

epochs = [i for i in range(num_epochs)]
fig , ax = plt.subplots(1,2)

fig.set_size_inches(20,6)

ax[0].plot(epochs , train_loss_lst , label = 'Training Loss')
ax[0].plot(epochs , test_loss_lst , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , acc_train_lst , label = 'Training Accuracy')
ax[1].plot(epochs , acc_test_lst , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()


# Do interpret
if not model_bert_lstm:
    pred = transformers.pipeline("text-classification", model=model, tokenizer=tokenizer, device=0, return_all_scores=True)
    explainer = shap.Explainer(pred)

    shap_values = explainer(train_sentences[:200])
    shap.plots.bar(shap_values[:, :, 0].mean(0))
    shap.plots.bar(shap_values[:, :, 1].mean(0))
    shap.plots.bar(shap_values[:, :, 2].mean(0))
    shap.plots.bar(shap_values[:, :, 3].mean(0))
    shap.plots.bar(shap_values[:, :, 4].mean(0))
    shap.plots.bar(shap_values[:, :, 5].mean(0))
