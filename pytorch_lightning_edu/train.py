import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *
from Dataset_preprocess import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=42)
    parser.add_argument('--data_path',
                        type=str,
                        default='./data',
                        help='where to prepare data')
    parser.add_argument('--max_epoch',
                        type=int,
                        default=10,
                        help='maximum number of epochs to train')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=-1,
                        help='number of available gpus')
    parser.add_argument('--mode',
                        type=str,
                        default='mecab',
                        choices=['mecab', 'only_korean'])
    parser.add_argument('--save_path',
                        type=str,
                        default='checkpoints',
                        help='where to save checkpoint files')
    parser.add_argument('--valid_size',
                        type=float,
                        default=0.1,
                        help='size of validation file')
    parser.add_argument('--max_seq_len',
                        type=int,
                        default=200,
                        help='maximum length of input sequence data')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size')
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)     #seed 고정 (reproductable)

    dm = NSMCDataModule(
        data_path=args.data_path,
        mode=args.mode,
        valid_size=args.valid_size,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )
    dm.prepare_data()
    dm.setup('fit')

    model = NSMCClassification()

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=args.save_path,
        filename='{epoch:02d}-{val_acc:.3f}',
        verbose=True,
        save_last=False,
        mode='max',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(      #max epoch=10 이지만 early stopping 가능
        monitor='val_acc',
        mode='max',
    )

    trainer = Trainer(
        max_epochs=args.max_epoch,
        accelerator='gpu',
        strategy="ddp",      #병렬 처리 방식. https://velog.io/@jaylnne/Pytorch-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%AA%A8%EB%8D%B8%EC%9D%98-%EC%84%B1%EB%8A%A5%EC%9D%84-%EA%B7%B9%EB%8C%80%ED%99%94%ED%95%98%EB%8A%94-7%EA%B0%80%EC%A7%80-%ED%8C%81 참조할것!
        devices=args.num_gpus,
        auto_select_gpus=True,
        callbacks=[checkpoint_callback, early_stopping],
    )
    trainer.fit(model, dm)

    #형태소 분석을 거치지 않고 함. 형태소 분석을 거치지 않은 게 결과가 더 좋았다고 함.


if __name__ == '__main__':
    main()