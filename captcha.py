import torch
import numpy as np


class Captcha(torch.nn.Module):
    def __init__(self, image_dim: tuple, output_length: int):
        super(Captcha, self).__init__()
        self.image_dim = image_dim
        self.output_length = output_length
        self.seq1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(128, 64, 3),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout2d(0.2)
        )
        self.seq2 = torch.nn.Sequential(
            torch.nn.Linear(64 * 24 * 164, 128),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, output_length * 11)
        )
        # self.classify_layer = torch.nn.Sequential(
        #     torch.nn.Softmax(),
        #     torch.nn.Linear(11, 1)
        # )

    def forward(self, data):
        x = self.seq1(data)
        # print(x.shape)
        x = x.view(-1, np.prod(x.shape[1:]) )
        x = self.seq2(x)
        #print(x.shape)
        x_parts = []
        for i in range(0, self.output_length * 11, 11):
            tmp = torch.nn.Softmax()(x[:, i:i+11])
            x_parts.append(tmp)

        #print(len(x_parts))
        x = torch.cat(x_parts, 1)
        #print(x.shape)
        x = x.reshape((x.shape[0], self.output_length, 11))
        return x