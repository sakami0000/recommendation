import torch
from torch import nn
import torch.nn.functional as F


class InteractingLayer(nn.Module):
    """An implementation of Interacting Layer used in AutoInt.
    This models the correlations between different feature fields by multi-head self-attention mechanism.

    Arguments
    ---------
        embedding_size: Integer. The size of input feature vector.
        att_embedding_size: Integer. The embedding size in multi-head self-attention network.
        head_num: Integer. The head number in multi-head self-attention network.
        use_res: Boolean. Whether or not use standard residual connections before output.
        dropout_p: Float in [0, 1). Dropout rate of multi-head self-attention.
    
    Input shape
    -----------
        3D tensor with shape `(batch_size, field_size, embedding_size)`.
    
    Output shape
    ------------
        3D tensor with shape `(batch_size, field_size, att_embedding_size * head_num)`.
    
    References
    ----------
        [Song W, Shi C, Xiao Z, et al.
         AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J].
         arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """
    def __init__(self, embedding_size, att_embedding_size=8, head_num=2, use_res=True, dropout_p=0.):
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        super(InteractingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.dropout_p = dropout_p

        self.w_query = nn.Linear(self.embedding_size, self.att_embedding_size * self.head_num)
        self.w_key = nn.Linear(self.embedding_size, self.att_embedding_size * self.head_num)
        self.w_value = nn.Linear(self.embedding_size, self.att_embedding_size * self.head_num)

        nn.init.xavier_uniform_(self.w_query.weight)
        nn.init.xavier_uniform_(self.w_key.weight)
        nn.init.xavier_uniform_(self.w_value.weight)

        if self.use_res:
            self.w_res = nn.Linear(self.embedding_size, self.att_embedding_size * self.head_num)
            nn.init.xavier_uniform_(self.w_res.weight)

        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                'Unexpected input dimension %d, expected to be 3 dimensions' % (x.dim()))
        batch_size = x.size(0)

        # (b, f, e) -> (b, f, a * h)
        querys = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        # (b * h, f, a)
        querys = torch.cat(querys.split(split_size=self.att_embedding_size, dim=2), dim=0)
        keys = torch.cat(keys.split(split_size=self.att_embedding_size, dim=2), dim=0)
        values = torch.cat(values.split(split_size=self.att_embedding_size, dim=2), dim=0)

        # (b * h, f, f)
        attention = torch.matmul(querys, keys.transpose(1, 2))
        attention = attention / torch.sqrt(self.embedding_size)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout_p)

        # (b * h, f, a)
        result = torch.matmul(attention, values)
        if self.use_res:
            result += self.w_res(x)

        # (b, f, a * h)
        result = torch.cat(result.split(split_size=batch_size, dim=0), dim=2)
        result = self.relu(result)

        return result
