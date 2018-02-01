#
# end_to_end.py
#
# Try the idea of tuning D score directly
#
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as Data

class Logstic_Regression(nn.Module):
    def __init__(self, in_dim, n_class):
        super(Logstic_Regression, self).__init__()
        self.logstic = nn.Linear(in_dim, n_class)

    def forward(self, x):
        out = self.logstic(x)
        return out


if __name__ == '__main__':

    batch_size = 128
    num_epoches = 30

 # year17 data
    with open('../data/processed/numpy/year17.pkl', 'rb') as pf:
        objs = pickle.load(pf)
        year17_lang_train_X = objs[0]
        year17_lang_train_y = objs[1]
        year17_lang_test_X = objs[2]
        year17_lang_test_y = objs[3]

        year17_meaning_train_X = objs[4]
        year17_meaning_train_y = objs[5]
        year17_meaning_test_X = objs[6]
        year17_meaning_test_y = objs[7]

        year17_train_X = np.concatenate((year17_meaning_train_X, year17_lang_train_X), axis=1)
        year17_train_y = np.column_stack((year17_meaning_train_y, year17_lang_train_y))
        year17_test_X = np.concatenate((year17_meaning_test_X, year17_lang_test_X), axis=1)
        year17_test_y = np.column_stack((year17_meaning_test_y, year17_lang_test_y))

        # 先转换成 torch 能识别的 Dataset
        # http://bit.ly/2r8KzMI
        labels = pd.get_dummies(year17_lang_train_y).values.argmax(1)

        torch_dataset = Data.TensorDataset(data_tensor=torch.from_numpy(year17_train_X).float(),
                                           target_tensor=torch.from_numpy(labels))
        # 把 dataset 放入 DataLoader
        loader = Data.DataLoader(
            dataset=torch_dataset,  # torch TensorDataset format
            batch_size=batch_size,  # mini batch size
            shuffle=True,           # 要不要打乱数据 (打乱比较好)
            num_workers=2,          # 多线程来读数据
        )


        model = Logstic_Regression(15, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        for epoch in range(num_epoches):
            print('epoch {}'.format(epoch + 1))
            running_loss = 0.0
            running_acc = 0.0
            for step, (batch_X, batch_y) in enumerate(loader):
                batch_X, batch_y = Variable(batch_X), Variable(batch_y)
                # forward
                out = model(batch_X)
                loss = criterion(out, batch_y)
                running_loss += loss.data[0] * batch_y.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == batch_y).sum()
                running_acc += num_correct.data[0]
                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step > 0 and step % 10 == 0:
                    print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                        epoch + 1, num_epoches, running_loss / (batch_size * step),
                        running_acc / (batch_size * step)))
