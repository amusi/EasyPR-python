#-- coding: utf-8 --
import os
from six.moves import cPickle as pickle
import random
from util.read_etc import *

whether_dataset = {}

total_list = []

labels = ['no', 'has']

cnt = 0
for i, label in enumerate(labels):
    for name in os.listdir('whether_car/' + label):
        record = {'name': name, 'label': i, 'subdir': label}
        total_list.append(record)

random.shuffle(total_list)
train_size = int(0.7 * len(total_list))
print(len(total_list[train_size:]))

print(total_list[train_size:])
with open('whether_list_train.pickle', 'wb') as f:
    pickle.dump(total_list[:train_size], f, 2)

with open('whether_list_val.pickle', 'wb') as f:
    pickle.dump(total_list[train_size:], f, 2)



