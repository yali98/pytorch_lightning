import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl


class NSMCClassification(pl.LightningModule):

    def __init__(self):
        super(NSMCClassification, self).__init__()

        # load pretrained koBERT
        self.bert = BertModel.from_pretrained('pretrained', output_attentions=True)

        # simple linear layer (긍/부정, 2 classes)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)       #classification이니까 linear층 추가
        self.num_classes = 2

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        h_cls = out['last_hidden_state'][:, 0]      # bert 거친 결과
        logits = self.W(h_cls)
        attn = out['attentions']

        return logits, attn

    def training_step(self, batch, batch_nb):
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label']

        # forward
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        loss = F.cross_entropy(y_hat, label.long())     #linear층의 loss 함수

        # logs
        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):      #validation도 여기서
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label']

        # forward
        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        loss = F.cross_entropy(y_hat, label.long())

        # accuracy
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)
        self.log('val_acc', val_acc, prog_bar=True)

        return {'val_loss': loss, 'val_acc': val_acc}