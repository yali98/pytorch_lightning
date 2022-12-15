import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from transformers import BertModel


class NSMCClassification(pl.LightningModule):

    def __init__(self):
        super(NSMCClassification, self).__init__()

        # load pretrained koBERT
        self.bert = BertModel.from_pretrained('pretrained', output_attentions=True)

        # simple linear layer (긍/부정, 2 classes)
        self.W = nn.Linear(self.bert.config.hidden_size, 2)       #classification이니까 linear층 추가
        self.num_classes = 2

    #==========================  forward ==============================
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

    # ==========================  training ==============================
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


    # ==========================  validation ==============================
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
        self.log('val_acc', val_acc, prog_bar=True)     #epoch 돌때마다 확인하기 위해서 (accuracy)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_end(self, outputs):       #validation 할때마다 확인하는 함수
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}




    # ==========================  test ==============================
    #test set으로 진행하는 것 당연히 loss 계산 빼고는 validation이랑 같음.
    def test_step(self, batch, batch_nb):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label']

        y_hat, attn = self.forward(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())
        test_acc = torch.tensor(test_acc)

        self.log_dict({'test_acc': test_acc})

        return {'test_acc': test_acc}

    def test_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}

        return {'avg_test_acc': tensorboard_logs}



    # ==========================  optimizer  ==============================
    #optimizer도 정해주고
    def configure_optimizers(self):
        #no_grad한 것들 제외한 모든 parameter들을 담는 방식.
        parameters = []
        for p in self.parameters():
            if p.requires_grad:
                parameters.append(p)
            else:
                print(p)

        optimizer = torch.optim.Adam(parameters, lr=2e-05, eps=1e-08)

        return optimizer

    # ==========================  dataset ==============================
    """
    def setup():
    그러나 가독성을 위해 따로 클래스를 만든다.
    파일 새로 만들겠음.
    """
