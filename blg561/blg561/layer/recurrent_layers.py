#Adapted from CS231n
from .layers_with_weights import LayerWithWeights
from copy import deepcopy
from abc import abstractmethod
import numpy as np


class RNNLayer(LayerWithWeights):
    """ Simple RNN Layer - only calculates hidden states """
    def __init__(self, in_size, out_size):
        """ RNN Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, out_size)
        self.Wh = np.random.rand(out_size, out_size)
        self.b = np.random.rand(out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None, 'dWh': None, 'db': None}
        
    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        next_h = np.tanh(self.b + prev_h.dot(self.Wh) + x.dot(self.Wx)) # ht = tanh(b + Wh.h + Wx.x) N*H = H + N*H H*H + N*D D*H
        cache = (x, prev_h, next_h)                                     # saved for backprop
        return next_h, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        self.x = x
        _h = h0
        h = np.zeros((x.shape[0], x.shape[1], h0.shape[1]))
        self.cache = []
        
        for i in range(x.shape[1]): # iterate T step
            _h, cache = self.forward_step(x[:, i, :], _h) # forward 1 step
            h[:,i,:] = _h # save hidden states
            self.cache.append(cache)
        return h
        
    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
        """
        # next_h = np.tanh(self.b + prev_h.dot(self.Wh) + x.dot(self.Wx))
        (x, prev_h, next_h) = cache                              # N*D - N*H - N*H
        
        dtanh = (1 - next_h**2) * dnext_h                        # deriv of tanh * dout(dnext) = tanh'
                
        #deriv self.b + prev_h.dot(self.Wh) + x.dot(self.Wx)
        
        dx = dtanh.dot(self.Wx.T)                                # x.dot(self.Wx)) -> dx = dtanh.Wx -> N*H H*D
        dprev_h = dtanh.dot(self.Wh.T)                           # prev_h.dot(self.Wh)-> dprev_h = dtanh.Wh -> N*H = N*H  H*H
        dWx = x.T.dot(dtanh)                                     # x.dot(self.Wx)) -> dWx = x.dtanh  D*H = D*N N*H
        dWh = prev_h.T.dot(dtanh)                                # prev_h.dot(self.Wh) -> dWh = prev_h.dtanh -> H*H = H*N N*H        
        db = dtanh.sum(axis = 0)                                 # sum in axis 0 dtanh
        
        return dx, dprev_h, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, H)
            dWh: gradients of weights Wh, of shape (H, H)
            db: gradients of bias b, of shape (H,)
            }
        """
        T = dh.shape[1]
        dx = np.zeros((dh.shape[0], dh.shape[1], self.x.shape[2]))
        dprev_h = 0
        dWx = 0
        dWh = 0
        db = 0
        for i in range(T):
            _dx, dprev_h, _dWx, _dWh, _db = self.backward_step(dh[:, T-i-1, :] + dprev_h, self.cache[T-i-1])#backward pass reverse order  
            dx[:,T-i-1,:] += _dx #sum them all
            dWx += _dWx
            dWh += _dWh
            db += _db
        dh0 = dprev_h
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db}
        
        
class LSTMLayer(LayerWithWeights):
    """ Simple LSTM Layer - only calculates hidden states and cell states """
    def __init__(self, in_size, out_size):
        """ LSTM Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 4 * out_size)
        self.Wh = np.random.rand(out_size, 4 * out_size)
        self.b = np.random.rand(4 * out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None}
        
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))     
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def dtanh(self, x):
        return 1 - x**2    
    
    def forward_step(self, x, prev_h, prev_c):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
            prev_c: previous cell state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            next_c: next cell state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        a = self.b + prev_h.dot(self.Wh) + x.dot(self.Wx)                          # N*4H = 4H + N*H H*4H + N*D D*4H
        
        l = int(a.shape[1] / 4)
        
        #splits [inp, forget, output, inp_gate] and used activation functions
        inp = self.sigmoid(a[:, :l])                                               # NH sigmoid(input)
        forget = self.sigmoid(a[:, l:2*l])                                         # NH sigmoid(forget)
        output = self.sigmoid(a[:, 2*l:3*l])                                       # NH sigmoid(output)
        inp_gate = np.tanh(a[:, 3*l:])                                             # NH tanh(input gate)
        
        next_c = forget * prev_c + inp * inp_gate                                  # forget*prev_c + input*input_gate
        next_h = output * np.tanh(next_c)                                          # output * tanh(nextc)
        
        cache = (x, prev_h, prev_c, output, next_c, inp, inp_gate, forget, output) # saved for backprob
        
        return next_h, next_c, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Cell state should be initialized to 0.
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        self.x = x
        _h = h0
        h = np.zeros((x.shape[0], x.shape[1], h0.shape[1]))
        self.cache = []
        _c = 0
        for i in range(x.shape[1]):
            _h, _c, cache = self.forward_step(x[:, i, :], _h, _c) # T step forward
            h[:, i, :] = _h # save hidden states
            self.cache.append(cache)
        return h
        
    def backward_step(self, dnext_h, dnext_c, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            dnext_c: gradient of loss with respect to
                     cell state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dprev_c: gradients of previous cell state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
        """
        (x, prev_h, prev_c, output, next_c, inp, inp_gate, forget, output) = cache
        
        #derivatives of all gates [inp, forget, output, inp_gate] and next_c
        doutput = dnext_h * np.tanh(next_c)                             # output * tanh(nextc) --> dout = dnext_h * tanh(nextc)
        doutput = doutput * output * (1 - output)                       # output = sigmoid(output) deriv -> dout * dsigmoid(output)
        
        dnextc = dnext_h * output * (1 - np.tanh(next_c)**2) + dnext_c  # dnextc = dnexth * output * (tanh')
      
        dinp_gate = dnextc * inp                                        # dinput_gate = dnextc * input
        dinp_gate = dinp_gate * (1-inp_gate**2)                         # dinput_gate * tanh'
        
        dinput = dnextc * inp_gate                                      # dinput = dnextc * input_gate
        dinput = dinput * inp * (1-inp)                                 # dinput * dsigmoid(input)       
        
        dforget = dnextc * prev_c                                       # dforget = dnextc * prevc
        dforget = dforget * forget*(1-forget)                           # dforget * dsigmoid(forget)
        
        a = np.concatenate((dinput, dforget), axis=1)                   # concatenate all gates to reach a
        a = np.concatenate((a, doutput), axis=1)
        a = np.concatenate((a, dinp_gate), axis=1)

        #derivatives of a = self.b + prev_h.dot(self.Wh) + x.dot(self.Wx)
        dx = a.dot(self.Wx.T)                                           # x.Wx -> dx -> a.Wx
        dWx = x.T.dot(a)                                                # x.Wx -> dWx -> x.a 
        dWh = prev_h.T.dot(a)                                           # prevh.Wh -> dWh -> prevh.a
        dprev_h = a.dot(self.Wh.T)                                      # prevh.Wh -> dprevh -> a.Wh
        
        dprev_c = forget * dnextc                                       # dprevc = forget * dnextc
        db = a.sum(axis=0)                                              # sum of a

        return dx, dprev_h, dprev_c, dWx, dWh, db

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 4H)
            dWh: gradients of weights Wh, of shape (H, 4H)
            db: gradients of bias b, of shape (4H,)
            }
        """
        T = dh.shape[1]
        dx = np.zeros((dh.shape[0],dh.shape[1], self.x.shape[2]))
        dWx = 0
        dWh = 0
        db = 0
        dprev_h = 0
        dprev_c =0
        
        # backward step reverse order
        for i in range(T):
            _dx, dprev_h, dprev_c, _dWx, _dWh, _db = self.backward_step(dh[:,T-i-1,:] +dprev_h, dprev_c, self.cache[T-i-1])             
            dx[:,T-i-1,:] += _dx
            dWx += _dWx
            dWh += _dWh
            db += _db
        dh0 = dprev_h
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db}
        

class GRULayer(LayerWithWeights):
    """ Simple GRU Layer - only calculates hidden states """
    def __init__(self, in_size, out_size):
        """ GRU Layer constructor
        Args:
            in_size: input feature dimension - D
            out_size: hidden state dimension - H
        """
        self.in_size = in_size
        self.out_size = out_size
        self.Wx = np.random.rand(in_size, 2 * out_size)
        self.Wh = np.random.rand(out_size, 2 * out_size)
        self.b = np.random.rand(2 * out_size)
        self.Wxi = np.random.rand(in_size, out_size)
        self.Whi = np.random.rand(out_size, out_size)
        self.bi = np.random.rand(out_size)
        self.cache = None
        self.grad = {'dx': None, 'dh0': None, 'dWx': None,
                     'dWh': None, 'db': None, 'dWxi': None,
                     'dWhi': None, 'dbi': None}
        
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))     
    
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def dtanh(self, x):
        return 1 - x**2 
    
    def forward_step(self, x, prev_h):
        """ Forward pass for a single timestep
        Args:
            x: input, of shape (N, D)
            prev_h: previous hidden state, of shape (N, H)
        Returns:
            next_h: next hidden state, of shape (N, H)
            cache: Values necessary for backpropagation, tuple
        """
        a = self.b + prev_h.dot(self.Wh) + x.dot(self.Wx) # 2H + N*H H*2H + N*D D*2H
        
        # splits [update, reset] and passed activation functions
        
        l = int(a.shape[1]/2)
        update = self.sigmoid(a[:,:l]) # sigmoid(update)
        reset = self.sigmoid(a[:,l:]) # sigmoid(reset)
        
        # hcandidate  calculated
        h_candidate = np.tanh(self.bi + (reset * prev_h).dot(self.Whi) + x.dot(self.Wxi))
        
        #next_h calculated
        next_h = update * prev_h + (1 - update) * h_candidate
        #h(t) = update  h(t1) + (1*update)*hcandidate
        cache = (x, prev_h, update, h_candidate, reset)
        
        return next_h, cache

    def forward(self, x, h0):
        """ Forward pass for the whole data sequence (of length T) of size minibatch N
        Values necessary in backpropagation need to be kept in self.cache as a list
        Args:
            x: input, of shape (N, T, D)
            h0: initial hidden state, of shape (N, H)
        Returns:
            h: hidden states of whole sequence, of shape (N, T, H)
        """
        self.x = x
        _h = h0
        _c = 1
        h = np.zeros((x.shape[0], x.shape[1], h0.shape[1]))
        
        self.cache = []        
        for i in range(x.shape[1]):
            _h, cache = self.forward_step(x[:,i,:], _h) # T step forward
            h[:,i,:] = _h #save hidden states
            self.cache.append(cache)
        return h
        
    def backward_step(self, dnext_h, cache):
        """ Backward pass for a single timestep
        Args:
            dnext_h: gradient of loss with respect to
                     hidden state, of shape (N, H)
            cache: necessary values from last forward pass
        Returns:
            dx: gradients of input, of shape (N, D)
            dprev_h: gradients of previous hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 2H)
            dWh: gradients of weights Wh, of shape (H, 2H)
            db: gradients of bias b, of shape (2H,)
            dWi: gradients of weights Wxi, of shape (D, H)
            dWhi: gradients of weights Whi, of shape (H, H)
            dbi: gradients of bias bi, of shape (H,)
        """
        (x, prev_h, update, h_candidate, reset) = cache
        
        """
        a = self.b + prev_h.dot(self.Wh) + x.dot(self.Wx)    
        l = int(a.shape[1]/2)
        update = self.sigmoid(a[:,:l]) # sigmoid(update)
        reset = self.sigmoid(a[:,l:]) # sigmoid(reset)        
        h_candidate = np.tanh(self.bi + (reset * prev_h).dot(self.Whi) + x.dot(self.Wxi))        
        next_h = update * prev_h + (1 - update) * h_candidate
        """
        
        #derivative of update * prev_h + (1 - update) * h_candidate -> dnext_h *
        dprev_h1 = update * dnext_h                          # dprevh => update
        dupdate = prev_h * dnext_h - h_candidate * dnext_h   # dupdate => prevh - hcandidate      
        dh_candidate = (1 - update) * dnext_h                # dhcandidate => 1-update
        dupdate = dupdate * (update * (1 - update))          # derivative of sigmoid(update)
        
        
        #derivative of np.tanh(self.bi + (reset * prev_h).dot(self.Whi) + x.dot(self.Wxi)) -> dh_candidate .dot
        dh_candidate = dh_candidate * (1 - h_candidate**2)   # deriv of tanh
        dx1 = dh_candidate.dot(self.Wxi.T)                   # dx => x.dot(self.Wxi) -> Wxi
        dWxi = x.T.dot(dh_candidate)                         # dWxi => x.dot(self.Wxi) -> x
        dWhi = (prev_h * reset).T.dot(dh_candidate)          # dWhi => (reset * prev_h).dot(self.Whi) -> (reset * prev_h)
        dresetprev = dh_candidate.dot(self.Whi.T)            # dresetprev => (reset * prev_h).dot(self.Whi) ->  Whi  
        dbi = np.sum(dh_candidate, axis = 0)                 # sum of all dhcandidate in axis 0   
        
        #derivative of dresetprev, split dprev and dreset + 
        dprev_h2 = dresetprev * reset
        dreset = dresetprev * prev_h
        dreset = dreset * (reset * (1 - reset))              # dreset -> deriv of sigmoid(reset)
                
        #derivative of self.b + prev_h.dot(self.Wh) + x.dot(self.Wx) -> da.dot
        da = np.concatenate((dupdate, dreset), axis=1)       # concat [update, reset]
        dx2 = da.dot(self.Wx.T)                              # dx => x.dot(self.Wx) -> Wx
        dWx = x.T.dot(da)                                    # dWx => x.dot(self.Wx) -> x
        dWh = prev_h.T.dot(da)                               # dWh => prev_h.dot(self.Wh) -> prev_h
        dprev_h3 = da.dot(self.Wh.T)                         # dprev_h => prev_h.dot(self.Wh) -> Wh
        db = np.sum(da, axis=0)                              # sum of all da in axis 0  
        
        
        dx = dx1 + dx2 #sum all used xs
        dprev_h = dprev_h1 + dprev_h2 + dprev_h3 #sum all used prevhs
        
        
        return dx, dprev_h, dWx, dWh, db, dWxi, dWhi, dbi

    def backward(self, dh):
        """ Backward pass for whole sequence
        Necessary data for backpropagation should be obtained from self.cache
        Args:
            dh: gradients of all hidden states, of shape (N, T, H)
        Calculates gradients and saves them to the dictionary self.grad
        self.grad = {
            dx: gradients of inputs, of shape (N, T, D)
            dh0: gradients of initial hidden state, of shape (N, H)
            dWx: gradients of weights Wx, of shape (D, 2H)
            dWh: gradients of weights Wh, of shape (H, 2H)
            db: gradients of bias b, of shape (2H,)
            dWxi: gradients of weights Wx, of shape (D, H)
            dWhi: gradients of weights Wh, of shape (H, H)
            dbi: gradients of bias b, of shape (H,)
            }
        """
        T = dh.shape[1]
        dx = np.zeros((dh.shape[0],dh.shape[1], self.x.shape[2]))
        db = 0
        dbi = 0
        dWx = 0
        dWh = 0
        dWxi = 0
        dWhi = 0
        dprev_h = 0
        for i in range(T):
            # backward step reverse order
            _dx, dprev_h, _dWx, _dWh, _db, _dWxi, _dWhi, _dbi = self.backward_step(dh[:,T-i-1,:] +dprev_h, self.cache[T-i-1])            
            dx[:,T-i-1,:] += _dx
            dWx += _dWx
            dWh += _dWh
            db += _db
            dWxi += _dWxi
            dWhi += _dWhi
            dbi += _dbi
        dh0 = dprev_h
        self.grad = {'dx': dx, 'dh0': dh0, 'dWx': dWx, 'dWh': dWh, 'db': db, 'dWxi': dWxi, 'dWhi': dWhi, 'dbi': dbi}

