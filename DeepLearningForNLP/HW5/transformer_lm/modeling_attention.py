'''
import torch.nn as nn
'''
import sys
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden):
        """Self-attention module which computes softmax(xQ @ xK^T) @ xV

        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
        """
        # Task 1.1: Create layers requires for self-attention (1 point)
        # YOUR CODE STARTS HERE (~4 lines)
        super().__init__()
        self.q = nn.Linear(input_size,hidden)
        self.k = nn.Linear(input_size,hidden)
        self.v = nn.Linear(input_size,hidden)
        self.scale = hidden ** 0.5 
        

        # YOUR CODE ENDS HERE

    def forward(self, x):
        """Softmax(xQ @ xK^T) @ xV

        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
        """
        # Task 1.2: Compute Self Attention (3 points)
        # 1. Compute key, query and value matrices from your input x
        # 2. Compute the scores using query and key matrices
        # 3. Compute probabilities using softmax and scale the scores using
        # 4. Compute the output using probabilities and value matrices
        #
        # Write shape of each tensor for each line of code
        # for example:
        #       Suppose batch_size = 3 and seq_len = 5
        #       x = torch.zeros(3, 5) # shape [batch_size, seq_len]
        #       x = x.unqueeze(1)     # shape [batch_size, 1, seq_len]
        #
        # NOTE: Remmenber that we work with batches of data [batch_size, seq_len, hidden],
        # not just single examples [seq_len, hidden] as we did in the lecture. This changes your shapes a bit.
        #
        # YOUR CODE STARTS HERE (~ can be implemented in 4 lines or 3 if you combine steps 2 and 3 into one operation)
        Q,K,V = self.q(x), self.k(x), self.v(x) # Compute K,Q,V
        scores  = torch.bmm(Q ,K.transpose(1,2)) / self.scale  # Compute attention scores and scale those scores
        probs= torch.softmax(scores,dim=-1) # Applying softamx to get attention weights 
        return torch.bmm(probs , V) # MatrixMul of probs and V to get weighted sum
        # Can reduce this to 4 lines if use self.(q,k,v) instead of declaring in first line 
        '''
        Add comments on each line refercning shape before submisson
        '''
        # YOUR CODE ENDS HERE

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_size, hidden, num_heads, causal=False, dropout=0):
        """
        Args:
            input_size: int, size of input vectors
            hidden: int, size of output vectors and hidden states
            num_heads: int, number of attention heads, should be a divisor of hidden
            causal: use causal masking (do not allow queires to look to the keys that correspond to the future tokens)
        """
        if hidden % num_heads:
            raise ValueError(f"hidden should be divisible by num_heads, "
                             f"but got hidden={hidden} and num_heads={num_heads}")
        super().__init__()

        self.k = nn.Linear(input_size, hidden)
        self.q = nn.Linear(input_size, hidden)
        self.v = nn.Linear(input_size, hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

        self.num_heads = num_heads
        self.head_size = hidden // num_heads
        self.scale = self.head_size ** 0.5
        self.causal = causal  # causal masking

    def forward(self, x, return_attention=False):
        """Computes [Softmax(x Q_1 @ x K_1^T) @ x V_1 : ... : Softmax(x Q_heads @ x K_heads^T) @ x V_heads] @ U
        Args:
            x: FloatTensor[batch_size, seq_len, input_size]

        Returns:
            FloatTensor[batch_size, seq_len, hidden]
            if return_attention is True, returns also FloatTensor[batch_size * num_heads, seq_len, seq_len]
        """
        bs, seq, _ = x.shape

        # Task 2.1 (3 points)
        # YOUR CODE STARTS HERE (Our implementation is in 3 lines, one for each for k, q and v)
        Q,K,V = self.q(x), self.k(x), self.v(x) # [Batch size ,seq , hidden]
        # TODO: Check if this code was here before or added by me 
        # [bs, seq, hidden] -> [bs, seq, num_heads, head_size] -> [bs, num_heads, seq, head_size]
        Q,K,V = Q.view(bs,seq,self.num_heads,self.head_size).transpose(1,2) ,K.view(bs,seq,self.num_heads,self.head_size).transpose(1,2), V.view(bs,seq,self.num_heads,self.head_size).transpose(1,2)
        #Reshape to target shape [batch_size * num_heads, seq_len, head_size]
        Q,K,V = Q.reshape(bs*self.num_heads,seq,self.head_size), K.reshape(bs*self.num_heads,seq,self.head_size), V.reshape(bs*self.num_heads,seq,self.head_size)
        scores = torch.matmul(Q,K.transpose(-2,-1)/ self.scale) # Q*K^T / scale , shape = [bs * num_heads, seq, seq]

                


        # Target shape  = batch * num_heads, seq , hidden /num_head
        # YOUR CODE ENDS HERE

        if self.causal:
            # Task 2.2 (1 point)
            # Apply casual mask to the scores
            # YOUR CODE STARTS HERE (Our implementation is in 2 lines)
            causal_mask = torch.triu(torch.ones(seq,seq, device=scores.device), diagonal=1) # shape of [seq,seq]
            scores = scores.masked_fill_(causal_mask ==1 , float('-inf')) # shape = [bs * num_heads, seq, seq]

            # YOUR CODE ENDS HERE

        # Task 2.3 (2 points)
        # Compute probability (probs) and attention (att), remember to apply mixing matrix
        # YOUR CODE STARTS HERE (can be implemented in 4 lines)
        probs = self.dropout(torch.softmax(scores,dim=-1)) # shape = [bs * num_heads, seq, seq]
        att = torch.matmul(probs,V)#  shape =[bs * num_heads, seq, head_size]
        att = att.reshape(bs,self.num_heads,seq,self.head_size).transpose(1,2).reshape(bs,seq,self.num_heads * self.head_size) # [bs , seq, hidden]
        # YOUR CODE ENDS HERE

        if return_attention:
            return att, probs

        return att