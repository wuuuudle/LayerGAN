import json
import numpy as np
from functools import reduce
import tensorflow as tf


# 生成字典
def generate_dic(txt):
    dic = set()
    for chr in txt:
        dic.add(chr)
    dic_out = {}
    for i, chr in enumerate(dic):
        dic_out[chr] = i + 1
    json.dump(dic_out, open('dic.json', 'w', encoding='utf8'))


dic = json.load(open('dic.json', encoding='utf8'))


def char2int(char):
    return dic[char]


def str2int(text):
    return [char2int(chr) for chr in text]


def int2char(index: int):
    for i in dic:
        if index == dic[i]:
            return i
    return 'null'


class DataLoader():
    def __init__(self, vocab_size, max_seq_len, path):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.path = path
        self.data = []
        self.x = []
        self.y = []
        self.split()
        self.index = 0

    def padding(self, inp):
        return inp + [0 for _ in range(self.max_seq_len - len(inp))]

    def split(self):
        with open(self.path, encoding='utf8') as f:
            text = f.read()
            # a = [chr + '。' for chr in text.split('。')]
            # lens = [len(chr) for chr in a]
            # sentence_list = []
            # i = 0
            # while i < len(a):
            #     j = i + 1
            #     while True:
            #         if reduce(lambda x, y: x + y, lens[i:j]) > 500:
            #             sentence_list.append(str2int(''.join(a[i:j - 1])))
            #             break
            #         j += 1
            #         if j > len(a):
            #             break
            #     i = j
            sentence_list = [str2int(i) for i in text.split('\n\n')]
        print('Loading data...')
        for num, sen in enumerate(sentence_list):
            for i in range(len(sen)):
                self.x.append(self.padding([self.vocab_size - 1] + sen[:i]))
                self.y.append([sen[i]])
            print('\r%d/%d' % (num, len(sentence_list) - 1), end='')
        self.x = np.array(self.x)
        self.y = np.array(self.y)
        # lens = len(self.text)
        # size_t = lens // 500
        # for i in range(size_t):
        #     self.data.append(str2int(self.text[i * 500: (i + 1) * 500]))

    def get_batch(self, batch_size):
        # target = np.zeros(shape=(batch_size, self.max_seq_len))
        inp = np.zeros(shape=(batch_size, self.max_seq_len), dtype=np.int)
        if self.index + batch_size < len(self.data):
            target = np.array(self.data[self.index:self.index + batch_size])
        else:
            target = np.array(self.data[self.index:] + self.data[:(self.index + batch_size) % len(self.data)])
        inp[:, 0] = self.vocab_size - 1
        inp[:, 1:] = target[:, :self.max_seq_len - 1]
        return inp, target

    def read_gen_all(self):
        return self.x, self.y

    def read_dis_data(self, gen: tf.keras.Model):
        print('generate_fake_data')
        lens = len(self.x)
        sizes = lens // 10240
        y_fake = []
        for i in range(sizes):
            out = gen.predict(self.x[i * 10240: (i + 1) * 10240, :])
            [y_fake.append([tf.argmax(o)]) for o in out]
            print('\r%d/%d' % (i, sizes - 1), end='')
        out = gen.predict(self.x[sizes * 10240:, :])
        [y_fake.append([tf.argmax(o)]) for o in out]
        true = []
        fake = []
        a = (self.x == 0).argmax(axis=1)
        sizes = len(a)
        for i, v in enumerate(a):
            temp = self.x[i].copy()
            temp[v] = self.y[i][0]
            true.append(temp)
            temp = self.x[i].copy()
            temp[v] = y_fake[i][0]
            fake.append(temp)
            print('\r%d/%d' % (i, sizes - 1), end='')
        return np.array(true), np.array(fake)

    def rollout_reward(self, gen: tf.keras.Model, dis: tf.keras.Model, candidate):
        print('generate_candidate_data...')
        lens = len(self.x)
        sizes = lens // 10240
        new_input_x = []
        new_input_y = []
        reward = []
        for i in range(sizes):
            out = gen.predict(self.x[i * 10240: (i + 1) * 10240, :])
            index = np.argsort(out)
            index = index[:, ::-1]  # 排序由概率由大到小
            for j in range(candidate):
                [new_input_x.append(o) for o in self.x[i * 10240: (i + 1) * 10240, :]]
                [new_input_y.append([o]) for o in index[:, j]]
            print('\r%d/%d' % (i, sizes - 1), end='')
        out = gen.predict(self.x[sizes * 10240:, :])
        index = np.argsort(out)
        index = index[:, ::-1]  # 排序由概率由大到小
        for j in range(candidate):
            [new_input_x.append(o) for o in self.x[sizes * 10240:, :]]
            [new_input_y.append([o]) for o in index[:, j]]

        print('generate_reward_data...')
        new_input_x = np.array(new_input_x)
        new_input_y = np.array(new_input_y)
        I = []
        a = (new_input_x == 0).argmax(axis=1)
        for i, v in enumerate(a):
            temp = new_input_x[i]
            temp[v] = new_input_y[i][0]
            I.append(temp)
        I = np.array(I)
        lens = len(self.x)
        sizes = lens // 10240
        for i in range(sizes):
            print('\r%d/%d' % (i, sizes - 1), end='')
            out = dis.predict(I[i * 10240:(i + 1) * 10240, :])[:, 0]
            [reward.append(o) for o in out]
        out = dis.predict(I[sizes * 10240:, :])[:, 0]
        [reward.append(o) for o in out]
        return new_input_x, new_input_y, np.array(reward)


if __name__ == '__main__':
    fin = open('wolalala.txt', 'r', encoding='utf8')
    text = fin.read()
    fin.close()
    generate_dic(text)
