import numpy as np
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

import sobamchan_chainer
import sobamchan_utility
utility = sobamchan_utility.Utility()

class LSTM(sobamchan_chainer.Model):
    
    def __init__(self, v, k):
        super(LSTM, self).__init__(
            embed = L.EmbedID(v, k),
            Wz = L.Linear(k, k),
            Wi = L.Linear(k, k),
            Wf = L.Linear(k, k),
            Wo = L.Linear(k, k),
            Rz = L.Linear(k, k),
            Ri = L.Linear(k, k),
            Rf = L.Linear(k, k),
            Ro = L.Linear(k, k),
            W = L.Linear(k, v),
        )

    def __call__(self, s):
        loss_sum = 0
        v, k = self.embed.W.data.shape
        h = Variable(np.zeros((1, k), dtype=np.float32))
        c = Variable(np.zeros((1, k), dtype=np.float32))
        for i in range(len(s)-1):
            next_word_id = s[i+1]
            tx = Variable(np.array([next_word_id], dtype=np.int32))
            x_k = self.embed(self.prepare_input([s[i]], dtype=np.int32))
            _z = self.Wz(x_k) + self.Rz(h)
            z = F.tanh(_z)
            _i = self.Wi(x_k) + self.Ri(F.dropout(h))
            i = F.sigmoid(_i)
            _f = self.Wf(x_k) + self.Rf(F.dropout(h))
            f = F.sigmoid(_f)
            c = i * z + f * c
            _o = self.Wo(x_k) + self.Ro(h)
            o = F.sigmoid(_o)
            y = h = o * F.tanh(c)
            loss = F.softmax_cross_entropy(self.W(h), tx)
            loss_sum += loss

        return loss_sum
