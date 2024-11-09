import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ShiftedSparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        self.group_size = config.group_size

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        qkv = self.qkv_proj(hidden_states)
        q, k, v = rearrange(qkv, 'b n (three h d) -> three b h n d', three=3, h=self.num_heads)
        
        attn_output = self.shifted_sparse_attention(q, k, v, attention_mask)
        
        return attn_output
    def shifted_sparse_attention(self, q, k, v, attention_mask=None):
      batch_size, num_heads, seq_len, head_dim = q.shape
      group_size = self.group_size
      
      # Ensure seq_len is divisible by group_size
      assert seq_len % group_size == 0, f"Sequence length {seq_len} must be divisible by group size {group_size}"
      num_groups = seq_len // group_size
  
      # Reshape q, k, v to (batch_size, num_groups, group_size, num_heads, head_dim)
      q = q.view(batch_size, num_heads, num_groups, group_size, head_dim).transpose(1, 2)
      k = k.view(batch_size, num_heads, num_groups, group_size, head_dim).transpose(1, 2)
      v = v.view(batch_size, num_heads, num_groups, group_size, head_dim).transpose(1, 2)
  
      # Split heads into two halves
      q1, q2 = q.chunk(2, dim=3)
      k1, k2 = k.chunk(2, dim=3)
      v1, v2 = v.chunk(2, dim=3)
  
      # Shift the second half
      q2 = torch.roll(q2, shifts=-group_size//2, dims=2)
      k2 = torch.roll(k2, shifts=-group_size//2, dims=2)
      v2 = torch.roll(v2, shifts=-group_size//2, dims=2)
  
      # Compute attention scores
      def attention(q, k, v, mask=None):
          scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
          if mask is not None:
              scores = scores.masked_fill(mask == 0, float('-inf'))
          attn_weights = F.softmax(scores, dim=-1)
          return torch.matmul(attn_weights, v)
  
      # Apply attention separately to each half
      out1 = attention(q1, k1, v1, attention_mask)
      out2 = attention(q2, k2, v2, attention_mask)
  
      # Concatenate the results
      out = torch.cat([out1, out2], dim=3)
  
      # Shift back the second half
      out[:, :, :, out.size(3)//2:] = torch.roll(out[:, :, :, out.size(3)//2:], shifts=group_size//2, dims=2)
  
      # Reshape back to original dimensions
      out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, num_heads * head_dim)
  
      return out

    def qkv_proj(self, x):
        return nn.Linear(self.hidden_size, 3 * self.hidden_size)(x)
