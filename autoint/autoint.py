import torch
from torch import nn

from layers import InteractingLayer


class AutoInt(nn.Module):
    """PyTorch implementation of AutoInt.

    Arguments
    ---------
        num_embeddings: Integer. The number of input features.
        embedding_dim: Integer. The size of input feature vector.
        att_layer_num: Integer. The Interacting Layer number to be used.
        att_embedding_size: Integer. The embedding size in multi-head self-attention network.
        att_head_num: Integer. The head number in multi-head self-attention network.
        att_use_res: Boolean. Whether or not use standard residual connections before output.
        att_dropout_p: Float in [0, 1). Dropout rate of multi-head self-attention.
        activation: {'sigmoid', 'linear'}. Output activation.

    References
    ----------
        [Song W, Shi C, Xiao Z, et al.
         AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J].
         arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """
    def __init__(self, num_embeddings, embedding_dim,
                 att_layer_num=3, att_embedding_size=8, att_head_num=2,
                 att_use_res=True, att_dropout_p=0., activation='sigmoid'):
        super(AutoInt, self).__init__()
        self.att_layer_num = att_layer_num
        self.activation = activation

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dropout = nn.Dropout2d(0.1)

        for i in range(self.att_layer_num):
            if i == 0:
                input_dim = embedding_dim
            else:
                input_dim = att_embedding_size * att_head_num
            
            setattr(self, f'interacting_layer_{i+1}',
                    InteractingLayer(input_dim,
                                     att_embedding_size=att_embedding_size,
                                     head_num=att_head_num,
                                     use_res=att_use_res,
                                     dropout_p=att_dropout_p))
        
        self.out = nn.Linear(att_embedding_size * att_head_num, 1)
        if activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.size(0)

        h_embedding = self.embedding(x)
        h_att = torch.unsqueeze(h_embedding.transpose(1, 2), 2)
        h_att = torch.squeeze(self.embedding_dropout(h_att)).transpose(1, 2)

        for i in range(self.att_layer_num):
            h_att = getattr(self, f'interacting_layer_{i+1}')(h_att)

        att_output = h_att.contiguous().view(batch_size, -1)
        out = self.out(att_output)
        if self.activation == 'sigmoid':
            out = self.activation(out)

        return out
