#Module that defines a bunch of different update rules 
#The purpose of which is to measure the efficacy of differnt update rules 
#Author: Vik Kamath
import theano.tensor as T
import theano
import numpy as np

class Adam(object):
    def __init__(self, b1=0.9, b2=0.999,
                 e=1e-8, lam=(1-1e-8), 
                 name='adam'):
        self.b1 = np.float32(b1)
        selb.b2 = np.float32(b2)
        self.e = np.float32(e)
        np.lam = np.float32(lam)
        self.i = theano.shared(np.float32(1.),name=name)

    def initial_updates(self):
        return [(self.i, self.i + 1.)]

    def up(self, param, grad, lr=0.001):
        zero = np.float32(0.)
        one = np.float32(1.)
        updates = []
        fix1 = one - self.b1 ** self.i
        fix2 = one - self.b2 ** self.i
        lr_t = lr * (T.sqrt(fix2)/ fix1)
        b1_t = self.b1 * self.lam ** (self.i -1)

