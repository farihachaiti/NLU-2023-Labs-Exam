import torch
import torch.utils.data as data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel
import os
import sys
import tensorflow as tf
sys.path.insert(0, os.path.abspath('../src/'))
from sklearn.metrics import classification_report
from transformers import Trainer, TrainingArguments
from transformers import default_data_collator, TFBertModel
import keras
import keras.utils
from keras import utils as np_utils
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt


class BERTModel(tf.keras.Model):

    def __init__(self, max_len, total_intent_no, total_slot_no, dropout_prob=0.5):
        super().__init__()

        self.max_len = max_len
        self.dropout_prob = dropout_prob
        self.total_intent_no = total_intent_no
        self.total_slot_no = total_slot_no
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = TFBertModel.from_pretrained("bert-base-uncased", max_length=self.max_len)
        self.slot_out = Dense(self.total_slot_no, activation='softmax')
        self.intent_out = Dense(self.total_intent_no, activation='softmax')
        # Dropout layer How do we apply it?
        self.dropout = Dropout(self.dropout_prob) #to prevent overfitting



    def tokenize(self, tokenizer, text_sequence, max_length):
        encoded = tokenizer(text_sequence, return_tensors='pt', is_split_into_words=True, max_length=self.max_len, padding='max_length', truncation=True)
        input_ids = encoded['input_ids'].unsqueeze(0)
        attention_mask = encoded['attention_mask'].unsqueeze(0)
        token_type_ids = encoded['token_type_ids'].unsqueeze(0)
        return input_ids, attention_mask, token_type_ids


    def call(self, inputs, **kwargs):
        outputs = self.model(inputs)
        slots = self.dropout(outputs[0])
        slots = self.slot_out(slots)
        intent = self.dropout(outputs[1])
        intent = self.intent_out(intent)

        return slots, intent