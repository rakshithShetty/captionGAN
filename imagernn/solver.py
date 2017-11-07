import time
import numpy as np
import theano
import theano.tensor as tensor
from imagernn.utils import numpy_floatX

class Solver:
  """
  solver worries about:
  - different optimization methods, updates, weight decays
  """
  def __init__(self,solver):
    if solver == 'rmsprop':
        self.build_solver_model = self.rmsprop
    elif solver == 'adam':
        self.build_solver_model = self.adam
    else:
        raise ValueError('ERROR: %s --> This solver type is not yet supported'%(solver))

  # Simply accumulate gradients. This can be used in inner loop to get the effect of a minibatch
  def accumGrads(self, tparams, grads, inp_list, cost, batch_size):

    accum_grads = [theano.shared(np.zeros_like(p.get_value()),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    agup = [(ag, ag + (1.0/batch_size) * g) for ag, g in zip(accum_grads, grads)]

    agclear = [(ag, ag.zeros_like()) for ag in accum_grads]

    f_grad_accum = theano.function(inp_list, cost,
                                    updates= agup,
                                    name='rmsprop_f_grad_shared')
    f_accumgrad_clr = theano.function([], [],updates=agclear, name='accum_grad_clr')

    return f_grad_accum, f_accumgrad_clr, accum_grads

# ========================================================================================
  def rmsprop(self, lr, tparams, grads, inp_list, cost, params, prior_updates=[], w_clip = None):
    clip = params['grad_clip']
    decay_rate = tensor.constant(params['decay_rate'], dtype=theano.config.floatX)
    smooth_eps = tensor.constant(params['smooth_eps'], dtype=theano.config.floatX)
    zipped_grads = [theano.shared(np.zeros_like(p.get_value()),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(np.zeros_like(p.get_value()),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    if clip > 0.0:
        rg2up = [(rg2, tensor.clip(decay_rate * rg2 + (1 - decay_rate) * (tensor.clip(g,-clip,clip) ** 2),0.0,np.inf))
             for rg2, g in zip(running_grads2, grads)]
    else:
        rg2up = [(rg2, tensor.clip(decay_rate * rg2 + (1 - decay_rate) * (g ** 2),0.0,np.inf))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp_list, cost,
                                    updates=zgup + rg2up + prior_updates,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, -lr * zg / (tensor.sqrt(rg2)+ smooth_eps))
                 for ud, zg, rg2 in zip(updir, zipped_grads,
                                            running_grads2)]
    if w_clip != None:
        print 'clipping weights with %.2f in RMS-PROP'%(w_clip)
        param_up = [(p, tensor.clip(p + udn[1], -w_clip, w_clip))
                    for p, udn in zip(tparams.values(), updir_new)]
    else:
        param_up = [(p, p + udn[1])
                    for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update, zipped_grads, running_grads2, updir

  def adam(self, lr, tparams, grads, inp, cost, params, prior_updates=[], w_clip = None):
      gshared = [theano.shared(p.get_value() * 0., name='%s_grad'%k) for k, p in tparams.iteritems()]
      gsup = [(gs, g) for gs, g in zip(gshared, grads)]

      f_grad_shared = theano.function(inp, cost, updates=gsup+prior_updates, profile=False)

      b1 = 0.1
      b2 = 0.001
      e = 1e-8

      updates = []

      i = theano.shared(np.float32(0.))
      i_t = i + 1.
      fix1 = 1. - b1**(i_t)
      fix2 = 1. - b2**(i_t)
      lr_t = lr * (tensor.sqrt(fix2) / fix1)

      for p, g in zip(tparams.values(), gshared):
          m = theano.shared(p.get_value() * 0.)
          v = theano.shared(p.get_value() * 0.)
          m_t = (b1 * g) + ((1. - b1) * m)
          v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
          g_t = m_t / (tensor.sqrt(v_t) + e)
          p_t = p - (lr_t * g_t)
          if w_clip != None:
            p_t = tensor.clip(p_t, -w_clip, w_clip)
            print 'clipping %s with %.2f'%(p.name, w_clip)
          updates.append((m, m_t))
          updates.append((v, v_t))
          updates.append((p, p_t))
      updates.append((i, i_t))

      f_update = theano.function([lr], [], updates=updates, on_unused_input='ignore', profile=False)

      return f_grad_shared, f_update, [], [], []
