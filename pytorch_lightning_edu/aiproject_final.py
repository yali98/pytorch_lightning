# -*- coding: utf-8 -*-
"""AIProject_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lSQP7VuXBhdA157no9q9Pme4gQnGlYy0
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import re
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator
from transformers import Seq2SeqTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ko-en")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-ko-en")

koen = pd.read_json('./data.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#koen.drop(columns = ['sn', 'data_set', 'subdomain','ko','mt','source_language','target_language','file_name','source','license','style','included_unknown_words','ner'], inplace = True)

import wandb
wandb.init(project="loss1, 5epoch, full data")

source_lang = "ko"
target_lang = "en"

import re
import sys

"""
    초성 중성 종성 분리 하기
	유니코드 한글은 0xAC00 으로부터
	초성 19개, 중성21개, 종성28개로 이루어지고
	이들을 조합한 11,172개의 문자를 갖는다.
	한글코드의 값 = ((초성 * 21) + 중성) * 28 + 종성 + 0xAC00
	(0xAC00은 'ㄱ'의 코드값)
	따라서 다음과 같은 계산 식이 구해진다.
	유니코드 한글 문자 코드 값이 X일 때,
	초성 = ((X - 0xAC00) / 28) / 21
	중성 = ((X - 0xAC00) / 28) % 21
	종성 = (X - 0xAC00) % 28
	이 때 초성, 중성, 종성의 값은 각 소리 글자의 코드값이 아니라
	이들이 각각 몇 번째 문자인가를 나타내기 때문에 다음과 같이 다시 처리한다.
	초성문자코드 = 초성 + 0x1100 //('ㄱ')
	중성문자코드 = 중성 + 0x1161 // ('ㅏ')
	종성문자코드 = 종성 + 0x11A8 - 1 // (종성이 없는 경우가 있으므로 1을 뺌)
"""
# 유니코드 한글 시작 : 44032, 끝 : 55199
BASE_CODE, CHOSUNG, JUNGSUNG = 44032, 588, 28

# 초성 리스트. 00 ~ 18
CHOSUNG_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

# 중성 리스트. 00 ~ 20
JUNGSUNG_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                 'ㅣ']

# 종성 리스트. 00 ~ 27 + 1(1개 없음)
JONGSUNG_LIST = [' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


def convert_ko(test_keyword):
    split_keyword_list = list(test_keyword)

    result = list()
    for keyword in split_keyword_list:
        # 한글 여부 check 후 분리
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - BASE_CODE

            char1 = int(char_code / CHOSUNG)
            # result.append(CHOSUNG_LIST[char1])

            char2 = int((char_code - (CHOSUNG * char1)) / JUNGSUNG)
            result.append(JUNGSUNG_LIST[char2])

            # char3 = int((char_code - (CHOSUNG * char1) - (JUNGSUNG * char2)))
            # if char3==0:
            #     result.append('#')
            # else:
            #     result.append(JONGSUNG_LIST[char3])

        else:
            result.append(keyword)
    # result
    return result


vowels = 'AEIOU'
consts = 'BCDFGHJKLMNPQRSTVWXYZ'
consts = consts + consts.lower()
vowels = vowels + vowels.lower()


def is_vowel(letter):
    return letter in vowels


def is_const(letter):
    return letter in consts


# get the syllables for vc/cv
def vc_cv(word):
    segment_length = 4  # because this pattern needs four letters to check
    pattern = [is_vowel, is_const, is_const, is_vowel]  # functions above
    split_points = []

    # find where the pattern occurs
    for i in range(len(word) - segment_length):
        segment = word[i:i + segment_length]

        # this will check the four letter each match the vc/cv pattern based on their position
        # if this is new to you I made a small note about it below
        if all([fi(letter) for letter, fi in zip(segment, pattern)]):
            split_points.append(i + int(segment_length / 2))

    # use the index to find the syllables - add 0 and len(word) to make it work
    split_points.insert(0, 0)
    split_points.append(len(word))
    syllables = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        syllables.append(word[start:end])
    return syllables


import eng_to_ipa as ipa

ipa_vowel = ["i", "y", "ɨ", "ʉ", "ɯ", "u", "ɪ", "ʏ", "ʊ", "e", "ø", "ɘ", "ɵ", "ɤ", "o", "ə", "ɛ", "œ", "ɜ", "ɞ", "ʌ",
             "ɔ", "æ", "ɐ", "a", "ɶ", "ɑ", "ɒ"]
double_vowel_ko = ["ㅖ", "ㅒ", "ㅕ", "ㅑ", "ㅠ", "ㅛ", "ㅟ", "ㅞ", "ㅙ", "ㅝ", "ㅘ"]
dict_double_one = {"ㅖ": "ㅔ", "ㅒ": "ㅐ", "ㅕ": "ㅓ", "ㅑ": "ㅏ", "ㅠ": "ㅜ", "ㅛ": "ㅗ", "ㅟ": "ㅣ", "ㅞ": "ㅔ", "ㅙ": "ㅐ", "ㅝ": "ㅓ",
                   "ㅘ": "ㅏ"}
dict_ipa_ko = {"i": "ㅣ", "y": "ㅣ", "ɨ": "ㅡ", "ʉ": "ㅡ", "ɯ": "ㅜ", "u": "ㅜ", "ɪ": "ㅣ", "ʏ": "ㅣ", "ʊ": "ㅜ", "e": "ㅔ",
               "ø": "ㅔ", "ɘ": "ㅓ", "ɵ": "ㅓ", "ɤ": "ㅗ", "o": "ㅗ", "ə": "ㅓ", "ɛ": "ㅔ", "œ": "ㅔ", "ɜ": "ㅓ", "ɞ": "ㅓ",
               "ʌ": "ㅓ", "ɔ": "ㅓ", "æ": "ㅐ", "ɐ": "ㅏ", "a": "ㅏ", "ɶ": "ㅓ", "ɑ": "ㅓ", "ɒ": "ㅗ"}


def convert_ipa2ko(word_list):
    result = []
    for word in word_list:
        isin = False
        for w in word:
            if w in ipa_vowel:
                result.append(dict_ipa_ko[w])
                isin = True
                break
        if not isin:
            result.append(" ")

    return result


def convert_ko_vowel(text_list):
    result = []
    for text in text_list:
        sentence = []
        for word in text:
            if word == " ": continue
            word = convert_ko(word)[0]
            if word in double_vowel_ko:
                word = dict_double_one[word]
            sentence += word

        result.append(sentence)

    return result


def convert_en_ko_vowel(text_list):
    result = []
    for text in text_list:
        sentence = []
        for word in text.split(" "):
            ipa_pro = vc_cv(word)
            temp = []
            for syll in ipa_pro:
                temp.append(ipa.convert(syll))
            ipa_pro = convert_ipa2ko(temp)
            sentence += ipa_pro

        result.append(sentence)
    return result


def find_similarity(text1, text2):
    count = 0
    if len(text1) <= len(text2):
        min = text1
        max = text2
    else:
        min = text2
        max = text1

    # print(min)
    # print(max)

    if len(min) == 0:
        return 1.0

    if len(min) < 2:
        if min in max:
            count += 1
    else:
        er = len(max) - len(min)
        temp = 0
        for i, mi in enumerate(min):
            for j, ma in enumerate(max[i:i + 1 + er]):
                if mi == ma:
                    count += 1
                    er - j
                    break

    # print(count)
    # print(len(min), len(max))
    return (len(min) - count) / len(min)


def find_similarity_batch(text1, text2):
    result = []
    temp1 = convert_ko_vowel(text1)
    temp2 = convert_en_ko_vowel(text2)

    for t1, t2 in zip(temp1, temp2):
        # t1 = hangul.sub('', t1)
        temp = 1.0
        try:
            temp = find_similarity(t1, t2)
        except:
            print(text1, temp2)
        result.append(temp)
    return result






def preprocess_function(examples):
    inputs = examples["ko_original"]
    inputcounts=[len(re.findall(r"[가-힣]", examples["ko_original"][i])) for i in range(len(examples["ko_original"]))]
    targets = examples[target_lang]
    targetscounts=examples["word_count_en"]
    
    
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    inputs_con=[]
    targets_con=[]

    for i in range(len(inputs)):
      inputs_con.append([inputcounts[i]]+model_inputs["input_ids"][i])
      #targets_con.append([targetscounts[i]]+model_inputs["labels"][i])
      #model_inputs["attention_mask"][i]=model_inputs["attention_mask"][i]+[1]

    model_inputs["input_ids"]=inputs_con
    return model_inputs

ds=Dataset.from_dict(koen)
splited_ds=ds.train_test_split(test_size=0.2)
tokenized_datasets = splited_ds.map(preprocess_function, batched=True)

from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="./test_trainer")

import syllables
print(syllables.estimate("hello_world,_it's_a_nice_day"))
class CustomTrainer(Seq2SeqTrainer):
  def compute_loss(self, model, inputs, return_outputs=False):
        target = [sentence.replace("▁", " ").strip() for sentence in tokenizer.batch_decode(inputs.get("input_ids")[:, 1:], skip_special_tokens=True)]
        ko_counts=inputs.get("input_ids")[:,0]
        inputs["input_ids"] = inputs.get("input_ids")[:,1:]
        to_input={"input_ids": inputs["input_ids"],"attention_mask": inputs["attention_mask"]}
        # forward pass
        outputs=model(**inputs)

        output_label = model.generate(**to_input)
        output_sentence=tokenizer.batch_decode(output_label, skip_special_tokens=True)
        en_count=[syllables.estimate(output_sentence[i]) for i in range(len(output_sentence))]

        #compute custom loss (suppose one has 3 labels with different weights)
        loss_add=ko_counts.cpu()-torch.Tensor(en_count).cpu()
        loss_add = loss_add.apply_(lambda x: abs(x))
        loss_add.requires_grad_()

        loss_add2 = torch.tensor(find_similarity_batch(target, output_sentence))
        loss_add2.requires_grad_()


        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # 모델에 입력을 넣어 output을 생성하고,
        outputs = model(**inputs)

        # 흠 나중에 고칠 필요가 있다고 하네요. 과거 state가 필요한 task에서 사용이 되는것 같습니다.
        # Save past state if it exists 
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0: 
            # 인덱스에 맞춰서 과거 ouput을 다 저장하고 있습니다. 어딘가에서 따로 사용이 되는듯합니다. 이번 포스팅에선 상관없는 부분!
            self._past = outputs[self.args.past_index]

        # label이 있으면 loss를 계산합니다.
        if labels is not None:
            # 기본적으로는 label_smoother 라는 loss를 사용하고 있습니다.
            loss = loss_add.mean()+self.label_smoother(outputs, labels)

        else:
            # 이부분은 사용되지 않는다고 하네요
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            
            loss = 5*loss_add2.mean() + abs(loss_add.mean()) * 4 / 10 + outputs["loss"] if isinstance(outputs, dict) else outputs[0] + loss_add2.mean() + abs(loss_add.mean()) * 4 / 10
            wandb.log({"semantic_loss": outputs["loss"], "morphological_loss1": abs(loss_add.mean()) * 4 / 10, "morphological_loss3": 5*loss_add2.mean(), "total_loss": loss})


        # 계산한 값들을 반환해 줍니다!
        return (loss, outputs) if return_outputs else loss

from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    fp16=True,
    report_to="wandb",
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

sample_text = "잘 번역되고 있나요?"
print(len(re.findall(r"[가-힣]", "잘 번역되고 있나요?")))
batch = tokenizer(sample_text, return_tensors="pt",padding=True)

batch.to('cuda')

generated_ids = model.cuda().generate(**batch)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
print(syllables.estimate(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]))