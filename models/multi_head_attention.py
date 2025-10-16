import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention', 'FANLayer']

class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component. If a string is passed,
            the corresponding activation from torch.nn.functional is used (default: 'gelu').
        use_p_bias (bool): If True, include bias in the linear transformations of p component (default: True).
            There is almost no difference between bias and non-bias in our experiments.
    """

    def __init__(self, input_dim, output_dim, p_ratio=0.1, activation='gelu', use_p_bias=True):
        super(FANLayer, self).__init__()

        # Ensure the p_ratio is within a valid range
        assert 0 < p_ratio < 0.5, "p_ratio must be between 0 and 0.5"

        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)
        g_output_dim = output_dim - p_output_dim * 2  # Account for cosine and sine terms

        # Linear transformation for the p component (for cosine and sine parts)
        self.input_linear_p = nn.Linear(input_dim, p_output_dim, bias=use_p_bias)

        # Linear transformation for the g component
        self.input_linear_g = nn.Linear(input_dim, g_output_dim)

        # Set the activation function
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

    def forward(self, src):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, input_dim).shape->(b,l,c)

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim), after applying the FAN layer.
        """

        # Apply the linear transformation followed by the activation for the g component
        g = self.activation(self.input_linear_g(src))

        # Apply the linear transformation for the p component
        p = self.input_linear_p(src)

        # Concatenate cos(p), sin(p), and activated g along the last dimension
        output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)

        return output
class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=None):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = FANLayer(in_features, in_features, use_p_bias=bias)
        self.linear_k = FANLayer(in_features, in_features, use_p_bias=bias)
        self.linear_v = FANLayer(in_features, in_features, use_p_bias=bias)
        self.linear_o = FANLayer(in_features, in_features, use_p_bias=bias)
        # # Explicitly initialize the bias terms to zero (if bias=True)
        # if self.bias:
        #     import torch.nn.init as init
        #     init.zeros_(self.linear_o.bias)
    def forward(self, q, k, v=None, attn_mask=None):
        if v is None:
            v = k
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)##(B,L,C)->(B*head,L,c/head)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, attn_mask)
        y = self._reshape_from_batches(y)#(B*head,L,c/head)->(B,L,C)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn_mask

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num #(B,L,C)->(B*head,L,c/head)
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )
