#-- coding: utf-8 --
import os
from six.moves import cPickle as pickle
import random
from util.read_etc import *

chars_dataset = {}

total_list = []

labels = os.listdir('chars/')
index2str = {}

cnt = 0
for label in labels:
    index2str[cnt] = label
    if label in province_mapping:
        index2str[cnt] = province_mapping[label]

    for name in os.listdir('chars/' + label):
        record = {'name': name, 'label': cnt}
        total_list.append(record)
    cnt += 1
random.shuffle(total_list)
train_size = int(0.7 * len(total_list))
print(len(total_list[train_size:]))

print(total_list[train_size:])
with open('chars_list_train.pickle', 'wb') as f:
    pickle.dump(total_list[:train_size], f, 2)

with open('chars_list_val.pickle', 'wb') as f:
    pickle.dump(total_list[train_size:], f, 2)

with open('index2str.pickle', 'wb') as f:
    pickle.dump(index2str, f, 2)


