"""

"""

import numpy as np
import tensorflow as tf


class BaseLayer(object):
    def __init__(self):
        pass

    def step_forward(self, x, prev_h):
        pass

    def step_backward(self, dnext_h, meta):
        pass

    def forward(self, x, h0):
        pass

    def backward(self, dh):
        pass


class LSTM(BaseLayer):
    def __init__(self, input_dim, h_dim, init_scale=0.02, name='lstm'):
        """
        In forward pass, please use self.params for the weights and biases for this layer
        In backward pass, store the computed gradients to self.grads
        - name: the name of current layer
        - input_dim: input dimension
        - h_dim: hidden state dimension
        - meta: to store the forward pass activations for computing backpropagation
        """
        self.name = name
        self.wx_name = name + "_wx"
        self.wh_name = name + "_wh"
        self.b_name = name + "_b"
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.params = {}
        self.grads = {}
        self.params[self.wx_name] = init_scale * np.random.randn(input_dim, 4 * h_dim)
        self.params[self.wh_name] = init_scale * np.random.randn(h_dim, 4 * h_dim)
        self.params[self.b_name] = np.zeros(4 * h_dim)
        self.grads[self.wx_name] = None
        self.grads[self.wh_name] = None
        self.grads[self.b_name] = None
        self.meta = None

    def step_forward(self, x, prev_h, prev_c):
        """
        x: input feature (N, D)
        prev_h: hidden state from the previous timestep (N, H)

        meta: variables needed for the backward pass
        """
        next_h, next_c, meta = None, None, None
        N, D = x.shape
        _, H = prev_h.shape

        # activation a: (N, 4H)
        a = x.dot(self.params[self.wx_name]) + \
            prev_h.dot(self.params[self.wh_name]) + \
            np.tile(self.params[self.b_name], [N, 1])

        # splite gates: (N, H)
        sigm = lambda x: 1 / (1 + np.exp(-x))
        i = sigm(a[:, 0: H])
        f = sigm(a[:, H: 2 * H])
        o = sigm(a[:, 2 * H: 3 * H])
        g = np.tanh(a[:, 3 * H:])

        # calculate cell
        next_c = f * prev_c + i * g
        z = np.tanh(next_c)
        next_h = o * z
        meta = [x, prev_h, prev_c, self.params[self.wx_name], self.params[self.wh_name],
                a, i, f, o, g, next_c, z, next_h]
        # return
        return next_h, next_c, meta

    def step_backward(self, dnext_h, dnext_c, meta):
        """
        dnext_h: gradient w.r.t. next hidden state
        meta: variables needed for the backward pass

        dx: gradients of input feature (N, D)
        dprev_h: gradients of previous hiddel state (N, H)
        dWh: gradients w.r.t. feature-to-hidden weights (D, H)
        dWx: gradients w.r.t. hidden-to-hidden weights (H, H)
        db: gradients w.r.t bias (H,)
        """
        dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None

        # update dnext_c
        x, prev_h, prev_c, Wx, Wh, a, i, f, o, g, next_c, z, next_h = meta
        dz = o * dnext_h
        do = z * dnext_h
        dnext_c += (1 - z ** 2) * dz

        # compute dgates from dnext_c
        df = dnext_c * prev_c
        dprev_c = dnext_c * f
        di = dnext_c * g
        dg = dnext_c * i

        # compute da
        da_i = i * (1 - i) * di
        da_f = f * (1 - f) * df
        da_o = o * (1 - o) * do
        da_g = (1 - g ** 2) * dg
        da = np.concatenate((da_i, da_f, da_o, da_g), axis=1)

        # compute dWh
        dprev_h = da.dot(self.params[self.wh_name].T)
        dWh = prev_h.T.dot(da)

        # compute dWx
        dx = da.dot(self.params[self.wx_name].T)
        dWx = x.T.dot(da)

        # compute db
        db = np.sum(da, axis=0)

        # return
        return dx, dprev_h, dprev_c, dWx, dWh, db

    def forward(self, x, h0):
        """
        Forward pass for an LSTM over an entire sequence of data. We assume an input
        sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
        size of H, and we work over a minibatch containing N sequences. After running
        the LSTM forward, we return the hidden states for all timesteps.

        Note that the initial hidden state is passed as input, but the initial cell
        state is set to zero. Also note that the cell state is not returned; it is
        an internal variable to the LSTM and is not accessed from outside.

        Inputs:
        - x: Input data of shape (N, T, D)
        - h0: Initial hidden state of shape (N, H)
        - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
        - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
        - b: Biases of shape (4H,)

        Returns:
        - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)

        Stores:
        - meta: Values needed for the backward pass.
        """
        h = None
        self.meta = []
        x = x.transpose(1, 0, 2)

        # extract dimension
        T, N, D = x.shape
        _, H = h0.shape

        # forward flow
        curr_h = h0
        curr_c = np.zeros([N, H])
        h = []
        for t in range(T):
            curr_h, curr_c, curr_meta = self.step_forward(x[t], curr_h, curr_c)
            h.append(curr_h)
            self.meta.append(curr_meta)
        h = np.array(h).transpose(1, 0, 2)

        # return
        return h

    def backward(self, dh):
        """
        Backward pass for an LSTM over an entire sequence of data.

        Inputs:
        - dh: Upstream gradients of hidden states, of shape (N, T, H)

        Returns a tuple of:
        - dx: Gradient of input data of shape (N, T, D)
        - dh0: Gradient of initial hidden state of shape (N, H)
        - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
        - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
        - db: Gradient of biases, of shape (4H,)
        """
        dx, dh0 = None, None
        dh = dh.transpose(1, 0, 2)

        # extact dimension
        T, N, H = dh.shape
        _, D = self.meta[0][0].shape  # x.shape

        # initialize gradients
        dx = np.zeros([T, N, D])
        dh0 = np.zeros([N, H])
        self.grads[self.wx_name] = np.zeros([D, 4 * H])
        self.grads[self.wh_name] = np.zeros([H, 4 * H])
        self.grads[self.b_name] = np.zeros([4 * H, ])

        # backward gradient flow accumulate
        dprev_h_t = np.zeros([N, H])  # first dprev is zeros
        dprev_c_t = np.zeros([N, H])
        for t in range(T - 1, -1, -1):
            dcurr_h = dh[t] + dprev_h_t  # accumulate d_h
            dcurr_c = dprev_c_t  # d_c is not accumulative since it
            dx_t, dprev_h_t, dprev_c_t, dWx_t, dWh_t, db_t = \
                self.step_backward(dnext_h=dcurr_h, dnext_c=dcurr_c, meta=self.meta[t])  # compute gradients in t-1
            dx[t] = dx_t
            self.grads[self.wx_name] += dWx_t
            self.grads[self.wh_name] += dWh_t
            self.grads[self.b_name] += db_t
        dh0 = dprev_h_t  # dh0 is h before first x
        dx = dx.transpose(1, 0, 2)
        self.meta = []

        # return
        return dx, dh0


class BiLSTM(object):
    pass