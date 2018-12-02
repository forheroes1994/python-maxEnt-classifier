from instence import *
from hyperparameter import Hyperparameter
import numpy as np
import sys
import re
import os
import random
import torch

torch.manual_seed(100)
random.seed(100)
np.random.seed(66)

class Classifier:
    def __init__(self):
        self.feature_alphabet = feature_alphabet()
        self.hyperparameter_1 = Hyperparameter()

    def clean_str(self,string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def read_file(self, path):
        Inst_list = []
        f = open(path, encoding = "UTF-8")
        for line in f.readlines():
            m_1 = inst()
            x = line.strip().split('|||')
            m_1.word = self.clean_str(x[0]).split(' ')
            # m_1.word = x[0].strip().split(' ')
            m_1.label = x[1].strip()
            Inst_list.append(m_1)
        f.close()
        return Inst_list

    def extract_sentence_feature_and_label_encoding(self,Inst_list):
        all_inst_feature = []
        for i in Inst_list:
            example = Example()
            for idx in range(len(i.word)):
                example.word_index.append('unigram = ' + i.word[idx])
            for idx in range(len(i.word) - 1):
                example.word_index.append('bigram = ' + i.word[idx] + '#' + i.word[idx + 1])
            for idx in range(len(i.word) - 2):
                example.word_index.append('trigram = ' + i.word[idx] + '#' + i.word[idx + 1] + '#' + i.word[idx + 2])
            if i.label == '0':
                example.label_index = [0, 0, 0, 0, 1]
                example.max_label_index = 4
            elif i.label == '1':
                example.label_index = [0, 0, 0, 1, 0]
                example.max_label_index = 3
            elif i.label == '2':
                example.label_index = [0, 0, 1, 0, 0]
                example.max_label_index = 2
            elif i.label == '3':
                example.label_index = [0, 1, 0, 0, 0]
                example.max_label_index = 1
            elif i.label == '4':
                example.label_index = [1, 0, 0, 0, 0]
                example.max_label_index = 0
            all_inst_feature.append(example)
        return all_inst_feature

    def creat_feature_alphabet(self, Inst_list):
        words = []
        for i in Inst_list:
            for j in i.word:
                words.append(j)
        featurealphabet = self.feature_alphabet.add_feature_alphabet(words)
        return featurealphabet

    def one_hot_encoding(self, train_Inst, dataset):
        one_hot_list = []
        all_Inst_feature = self.extract_sentence_feature_and_label_encoding(dataset)
        feat_alphabet = self.creat_feature_alphabet(train_Inst)
        for exam in all_Inst_feature:
            one_hot = Example()
            one_hot.label_index = exam.label_index
            one_hot.max_label_index = exam.max_label_index
            for j in exam.word_index:
                if j in feat_alphabet.dict:
                    one_hot.word_index.append(feat_alphabet.dict[j])
            one_hot_list.append(one_hot)
        return one_hot_list

    def Init_weight_array(self, train_Inst):
        feat_alphabet =self.creat_feature_alphabet(train_Inst)
        self.weight_array = np.random.rand(len(feat_alphabet.list), self.hyperparameter_1.class_num)
        return self.weight_array

    def get_max_index(self, result):
        max, index = result[0],0
        for idx in range(len(result)):
            if result[idx] > max:
                max, index = result[idx], idx
        return index

    def Y_list(self, one_hot_list):
        y_list = []
        for i in one_hot_list:
            sentence_result = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
            for j in i.word_index:
                sentence_result += np.array(self.weight_array[j])
            y_list.append(self.softmax((1/self.hyperparameter_1.batch_size)*sentence_result))
        return y_list

    def set_batchBlock(self, examples):
        if len(examples) % self.hyperparameter_1.batch_size == 0:
            batchBlock = len(examples) // self.hyperparameter_1.batch_size
        else:
            batchBlock = len(examples) // self.hyperparameter_1.batch_size + 1
        return batchBlock

    def start_and_end_pos(self, every_batchBlock, train_exam_list):
        start_pos = every_batchBlock * self.hyperparameter_1.batch_size
        end_pos = (every_batchBlock + 1) * self.hyperparameter_1.batch_size
        if end_pos >= len(train_exam_list):
            end_pos = len(train_exam_list)
        return start_pos, end_pos

    def count_loss(self, y):
        p = np.max(y)
        return -1 * np.log(p)

    def softmax(self, result):
        result_list = []
        bottom = 0
        max_idx = self.get_max_index(result)
        for index, value in enumerate(result):
            bottom += np.exp(value - result[max_idx])
        for index, value in enumerate(result):
            result_list.append(np.exp(value - result[max_idx])/bottom)
        return result_list

    def back_ward(self,list,result):
        label_list = []
        for i in list:
            label_list.append(i.label_index)
        pd_l_for_y = np.array(result)-np.array(label_list)
        for idx, val in enumerate(list):
            for j in val.word_index:
                self.init_grad_w[j] += pd_l_for_y[idx]
                self.weight_array[j] -= self.hyperparameter_1.lr * np.array(self.init_grad_w[j])

    def count_para(self,batch_list,batch_result):
        batch_corrects, batch_sum, batch_loss= 0, 0, 0
        for i,j in zip(batch_list,batch_result):
            if self.get_max_index(i.label_index) == self.get_max_index(j):
                batch_corrects += 1
            batch_sum += 1
            loss = self.count_loss(j)
        batch_loss += loss
        return batch_corrects,batch_sum,batch_loss

    def train(self,train_Inst, dev_Inst,test_Inst):
        train_exam_list = self.one_hot_encoding(train_Inst, train_Inst)
        dev_exam_list = self.one_hot_encoding(train_Inst, dev_Inst)
        test_exam_list = self.one_hot_encoding(train_Inst, test_Inst)
        train_size = len(train_exam_list)
        feat_alphabet = self.creat_feature_alphabet(train_Inst)
        self.weight_array = self.Init_weight_array(train_Inst)
        self.init_grad_w = np.zeros((len(feat_alphabet.list), self.hyperparameter_1.class_num))
        steps = 0
        for epoch in range(1,self.hyperparameter_1.epochs+1):
            steps +=1
            print("————第{}轮迭代，共{}轮————".format(epoch, self.hyperparameter_1.epochs))
            random.shuffle(train_exam_list)
            batchBlock = self.set_batchBlock(train_exam_list)
            all_corrects, all_accuracy, all_sum, all_loss = 0, 0, 0, 0
            for every_batchBlock in range(batchBlock):
                start_pos, end_pos = self.start_and_end_pos(every_batchBlock, train_exam_list)
                sentence_result = self.Y_list(train_exam_list[start_pos:end_pos])
                every_batch_corrects, every_batch_sum ,every_batch_loss = self.count_para(train_exam_list[start_pos:end_pos],sentence_result)
                self.back_ward(train_exam_list[start_pos:end_pos],sentence_result)
                self.init_grad_w = np.zeros(self.init_grad_w.shape)
                all_corrects += every_batch_corrects
                all_sum += every_batch_sum
                all_loss += every_batch_loss
            if steps % self.hyperparameter_1.log_interval == 0:
                accuracy = all_corrects / all_sum * 100.0
                sys.stdout.write(
                    '\rBatch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                                train_size,
                                                                                all_loss,
                                                                                accuracy,
                                                                                all_corrects,
                                                                                all_sum))
            if steps % self.hyperparameter_1.test_interval == 0:
                self.eval(dev_exam_list)


    def eval(self, dev_exam_list):
        corrects, accuracy, sum = 0, 0, 0
        sentence_result = self.Y_list(dev_exam_list)
        train_size = len(sentence_result)
        for idx in range(len(sentence_result)):
            num = self.get_max_index(sentence_result[idx])
            label_num = dev_exam_list[idx].max_label_index
            if num == label_num:
                corrects += 1
            sum += 1
        accuracy = corrects / sum * 100.0
        print('\nEvaluation -  acc: {:.4f}%({}/{}) \n'.format(accuracy,
                                                              corrects,
                                                              train_size))


a = Classifier()
train_Inst = a.read_file(path='data/raw.clean.train')
dev_Inst = a.read_file(path='data/raw.clean.dev')
test_Inst = a.read_file(path='data/raw.clean.test')
if os.path.exists("./Test_Result.txt"):
    os.remove("./Test_Result.txt")
    print("The 'Test_Result.txt' has been removed.")
    a.train(train_Inst, dev_Inst, test_Inst)
else:
    a.train(train_Inst, dev_Inst, test_Inst)



