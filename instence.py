from collections import OrderedDict
import pickle
class inst:
    def __init__(self):
        self.word = []
        self.label = ''

class all_inst:
    def __init__(self):
        self.all_word = []
        self.all_label = []

class feature_alphabet:
    def __init__(self):
        self.list = []
        self.dict = OrderedDict()

    def add_feature_alphabet(self, words):
        a = feature_alphabet()
        for idx in range(len(words)):
            a.list.append('unigram = ' + words[idx])
        for idx in range(len(words) - 1):
            a.list.append('bigram = ' + words[idx] + '#' + words[idx + 1])
        for idx in range(len(words) - 2):
            a.list.append('trigram = ' + words[idx] + '#' + words[idx + 1] + '#' + words[idx + 2])
        a.list = self.del_dup(a.list)
        for idx in range(len(a.list)):
            e = a.list[idx]
            a.dict[e] = idx
        return a

    def del_dup(self, list):
        if list:
            list.sort()
            last = list[-1]
            for i in range(len(list) - 2, -1, -1):
                if last == list[i]:
                    del list[i]
                else:
                    last = list[i]
        return list


    # def method3(self, list):
    #     temp = []
    #     [temp.append(i) for i in list if not i in temp]
    #     return temp

class Example:
    def __init__(self):
        self.word_index = []
        self.label_index = []
        self.max_label_index = 0
        self.max_label = 1

