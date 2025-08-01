import math
import copy
from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

# from ct_clip.mlm import MLM
# from ct_clip.visual_ssl import SimSiam, SimCLR

from transformers import BertTokenizer, BertModel

# NOTE: Import a Cross-Attn Module and PE Module
from positional_encodings.torch_encodings import PositionalEncoding3D, PositionalEncoding1D
from ct_clip.transformer_decoder import TransformerDecoder, TransformerDecoderLayer
# from transformer_decoder import TransformerDecoder, TransformerDecoderLayer

# NOTE

# helper functions

def identity(t, *args, **kwargs):
    return t

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

@contextmanager
def null_context():
    yield

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def cast_tuple(t):
    return t if isinstance(t, (tuple, list)) else (t,)

def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def l2norm(t):
    
    return F.normalize(t, dim = -1,eps=1e-6)

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)

# checkpointing helper function

def make_checkpointable(fn):
    @wraps(fn)
    def inner(*args):
        input_needs_grad = any([isinstance(el, torch.Tensor) and el.requires_grad for el in args])

        if not input_needs_grad:
            return fn(*args)

        return checkpoint(fn, *args)

    return inner

# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))

def group_dict_by_key(cond, d):
    return_val = [dict(),dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)

def string_begins_with(prefix, str):
    return str.startswith(prefix)

def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)

def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs

# helper classes

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w z) c -> b c h w z', h = h_r, w= w_r)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

# patch dropout

class PatchDropout(nn.Module):
    def __init__(self, prob):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob

    def forward(self, x, force_keep_all = False):
        if not self.training or self.prob == 0. or force_keep_all:
            return x

        b, n, _, device = *x.shape, x.device

        batch_indices = torch.arange(b, device = device)
        batch_indices = rearrange(batch_indices, '... -> ... 1')
        num_patches_keep = max(1, int(n * (1 - self.prob)))
        patch_indices_keep = torch.randn(b, n, device = device).topk(num_patches_keep, dim = -1).indices

        return x[batch_indices, patch_indices_keep]

# rotary positional embedding

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        inv_freq = self.inv_freq
        t = torch.arange(seq_len, device = device).type_as(inv_freq)
        freqs = torch.einsum('i , j -> i j', t, inv_freq)
        return torch.cat((freqs, freqs), dim = -1)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

def apply_rotary_pos_emb(freqs, t):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim = -1)

# transformer

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, causal = False, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias = False), LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None, rotary_pos_emb = None):
        h, device, scale = self.heads, x.device, self.scale

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        q = q * self.scale

        if exists(rotary_pos_emb):
            apply_rotary = partial(apply_rotary_pos_emb, rotary_pos_emb)
            q, k, v = map(apply_rotary, (q, k, v))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        attn = attn.type(sim.dtype)

        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_head = 64,
            heads = 8,
            causal = False,
            attn_dropout = 0.,
            ff_dropout = 0.,
            ff_mult = 4,
            checkpoint_during_training = False
    ):
        super().__init__()
        self.checkpoint_during_training = checkpoint_during_training

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, mult = ff_mult)),
            ]))

        self.norm_in = LayerNorm(dim)
        self.norm_out = LayerNorm(dim)

    def forward(
            self,
            x,
            rotary_pos_emb = None,
            mask = None
    ):
        can_checkpoint = self.training and self.checkpoint_during_training
        checkpoint_fn = make_checkpointable if can_checkpoint else identity

        x = self.norm_in(x)

        for attn, ff in self.layers:
            attn, ff = map(checkpoint_fn, (attn, ff))

            x = attn(x, mask, rotary_pos_emb) + x
            x = ff(x) + x

        return self.norm_out(x)

# text and vision transformers

class TextTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            num_tokens,
            max_seq_len,
            dim_head,
            rotary_pos_emb = None,
            causal = False,
            **kwargs
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if not rotary_pos_emb else None
        self.rotary_pos_emb = RotaryEmbedding(min(dim_head, 32)) if rotary_pos_emb else None

        self.cls_token = nn.Parameter(torch.randn(dim)) if not causal else None

        self.transformer = Transformer(dim, dim_head = dim_head, causal = causal, **kwargs)

    def forward(self, x, mask = None):
        b, n, device = *x.shape, x.device

        x = self.token_emb(x)

        if exists(self.abs_pos_emb):
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> 1 n d')

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            rotary_pos_emb = self.rotary_pos_emb(n + 1, device = device)

        if exists(self.cls_token):
            cls_tokens = repeat(self.cls_token, 'd -> b 1 d', b = b)
            x = torch.cat((cls_tokens, x), dim = 1)

            if exists(mask):
                mask = F.pad(mask, (1, 0), value = True)

        out = self.transformer(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
        return out

class VisionTransformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            image_size,
            patch_size,
            channels,
            patch_dropout = 0.5,
            **kwargs
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_tokens = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim)
        )

        self.pos_emb = nn.Embedding(num_patches, dim)
        self.patch_dropout = PatchDropout(patch_dropout)

        self.transformer = Transformer(dim, **kwargs)

        self.to_cls_tokens = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.Linear(dim, dim, bias = False),
            Rearrange('b d -> b 1 d')
        )

    def forward(
            self,
            x,
            keep_all_patches = False
    ):
        device = x.device

        x = self.to_tokens(x)
        b, n, _ = x.shape

        pos_emb = self.pos_emb(torch.arange(n, device = device))
        x = x + rearrange(pos_emb, 'n d -> 1 n d')

        x = self.patch_dropout(x, force_keep_all = keep_all_patches)

        out = self.transformer(x)

        cls_tokens = self.to_cls_tokens(out)
        return torch.cat((cls_tokens, out), dim = 1)

# contrastive learning functions

def model_forward_with_context(
        *,
        fn,
        args,
        freeze,
):
    encoding_context = null_context if not freeze else torch.no_grad

    with encoding_context():
        enc = fn(*args)

        if freeze:
            enc.detach_()

    return enc

# main clip class

class CTCLIP(nn.Module):
    def __init__(
            self,
            *,
            use_triplet_loss = 1,
            use_infoNCE_loss = 1,
            use_uncon_triplet_loss = 1,
            use_uncon_infoNCE_loss = 1,
            
            use_image2image_loss = 1,
            triplet_loss_margin = 0.1,
            positive_distance_threshold = 0.2,
            positive_threshold = 0.7,   # ON default, >0.7 can be positive
            negative_threshold = 1, # ON default, eveyone can be negative
            image_encoder = None,
            text_encoder = None,
            tokenizer = None,
            fusion_module = None,   # NOTE: Pass in a Transformer Decoder Module
            dim_text = 512,
            dim_image = 512,
            dim_latent = 512,
            num_text_tokens = 28897,
            text_enc_depth = 6,
            text_seq_len = 256,
            text_heads = 8,
            text_dim_head = 64,
            text_has_cls_token = False,
            text_pad_id = 0,
            text_rotary_pos_emb = False,
            text_causal_mask = False,
            text_eos_id = None,
            text_encode_without_mask = False,
            visual_enc_depth = 6,
            visual_heads = 8,
            visual_dim_head = 64,
            visual_image_size = 256,
            visual_patch_size = 32,
            visual_patch_dropout = 0.5,
            visual_has_cls_token = False,
            channels = 3,
            use_all_token_embeds = False,
            downsample_image_embeds = False,
            decoupled_contrastive_learning = False,
            extra_latent_projection = False,
            use_mlm = False,
            text_ssl_loss_weight = 0.05,
            use_visual_ssl = False,
            visual_ssl = None,
            visual_ssl_type = 'simsiam',
            visual_ssl_hidden_layer = -1,
            simclr_temperature = 0.1,
            image_ssl_loss_weight = 0.05,
            multiview_loss_weight = 0.1,
            checkpoint_during_training = False,
            **kwargs
    ):
        super().__init__()
        self.use_triplet_loss = use_triplet_loss
        self.triplet_loss_margin = triplet_loss_margin
        self.positive_distance_threshold = positive_distance_threshold  # triplet中确定ij>ik的阈值
        
        self.use_infoNCE_loss = use_infoNCE_loss
        self.positive_threshold = positive_threshold    # infoNCE中找到false negative
        self.negative_threshold = negative_threshold
        self.use_image2image_loss = use_image2image_loss
        self.use_uncon_triplet_loss = use_uncon_triplet_loss
        self.use_uncon_infoNCE_loss = use_uncon_infoNCE_loss
        #assert use_all_token_embeds or (visual_has_cls_token or text_has_cls_token), 'CLS token must be included on both vision and text transformers if you are not using fine-grained contrastive learning loss'
        self.dtype=torch.float32
        # store some parameters for access

        self.dim_text = dim_text
        self.dim_image = dim_image
        self.dim_latent = dim_latent

        self.image_channels = channels
        self.image_size = visual_image_size

        # instantiate text transformer

        self.text_pad_id = text_pad_id
        self.text_has_cls_token = text_has_cls_token
        self.text_seq_len = text_seq_len

        self.text_encode_without_mask = text_encode_without_mask # whether to pass in text mask to text encoder

        self.text_causal_mask = text_causal_mask
        self.text_eos_id = text_eos_id

        assert not (text_causal_mask and not exists(text_eos_id)), 'text EOS token id must be given if using causal mask in text transformer'

        if exists(text_encoder):
            self.text_transformer = text_encoder
        else:
            self.text_transformer = TextTransformer(
                dim = dim_text,
                num_tokens = num_text_tokens + (1 if use_mlm else 0),
                max_seq_len = text_seq_len,
                depth = text_enc_depth,
                heads = text_heads,
                causal = text_causal_mask,
                dim_head = text_dim_head,
                rotary_pos_emb = text_rotary_pos_emb,
                checkpoint_during_training = checkpoint_during_training
            )

        # instantiate image transformer

        self.visual_has_cls_token = visual_has_cls_token

        if exists(image_encoder):
            self.visual_transformer = image_encoder
        else:
            self.visual_transformer = VisionTransformer(
                dim = dim_image,
                image_size = visual_image_size,
                patch_size = visual_patch_size,
                channels = channels,
                depth = visual_enc_depth,
                heads = visual_heads,
                dim_head = visual_dim_head,
                patch_dropout = visual_patch_dropout,
                checkpoint_during_training = checkpoint_during_training
            )
        
        # NOTE: Pass in a Transformer Decoder Module and a Positional-Encoding Module
        if exists(fusion_module):
            self.fusion_module = fusion_module
        else:
            decoder_layer = TransformerDecoderLayer(d_model=dim_latent, nhead=8, normalize_before=True)
            decoder_norm = nn.LayerNorm(dim_latent)
            self.fusion_module = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)
        # self.pos_embedding = PositionalEncoding3D(dim_latent)(torch.zeros(1, 24, 24, 24, dim_latent)) # 1 H W D Dim
        self.pos_embedding = PositionalEncoding1D(dim_latent)(torch.zeros(1, 24, dim_latent)) # 1 D Dim
        # NOTE

        # text ssl

        self.use_mlm = use_mlm
        self.text_ssl_loss_weight = text_ssl_loss_weight if use_mlm else 0

        # image ssl

        self.use_visual_ssl = use_visual_ssl or exists(visual_ssl)
        self.image_ssl_loss_weight = image_ssl_loss_weight if use_visual_ssl else 0

        # text latent projection

        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias = False)

        # image latent projection

        if downsample_image_embeds:
            #assert use_all_token_embeds, 'must be using all token embeds for contrastive learning in order to downsampling'
            dim_conv=512
            self.to_visual_latent = nn.Sequential(
                RearrangeImage(),
                nn.Conv3d(dim_conv, dim_conv, 4, stride = 2, padding = 1, bias = False, groups = dim_conv),
                nn.Conv3d(dim_conv, dim_latent, 1),
                Rearrange('b c h w z -> b (h w z c)'),
                nn.Linear(dim_image, dim_latent, bias = False)
            )
        else:
            self.to_visual_latent = nn.Linear(dim_image, dim_latent, bias = False)

        # temperature

        self.temperature = nn.Parameter(torch.tensor(1.))

        # from https://arxiv.org/abs/2111.07783 (FILIP paper)
        self.use_all_token_embeds = use_all_token_embeds

        # proposed in https://arxiv.org/abs/2110.06848 (DCL) and https://arxiv.org/abs/2110.11316 (CLOOB)
        self.decoupled_contrastive_learning = decoupled_contrastive_learning

        self.multiview_loss_weight = multiview_loss_weight

        if tokenizer != None:
            self.tokenizer=tokenizer
        else:
            raise ValueError('Tokenzier is not defined.')
        
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)
        
    def freeze_text_encoder(self):
        for param in self.text_transformer.parameters():
            param.requires_grad = False
        for param in self.to_text_latent.parameters():
            param.requires_grad = False
            
    def open_partial_text_encoder(self):
        for name, param in self.text_transformer.named_parameters():
            if 'pooler' in name or 'encoder.layer.11' in name or 'encoder.layer.10' in name or 'encoder.layer.9' in name or 'encoder.layer.8' in name: # or 'encoder.layer.7' in name or 'encoder.layer.6' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.to_text_latent.parameters():
            param.requires_grad = True
            
    def open_text_encoder(self):
        for param in self.text_transformer.parameters():
            param.requires_grad = True
        for param in self.to_text_latent.parameters():
            param.requires_grad = True
        
    def freeze_vision_encoder(self):
        for param in self.visual_transformer.parameters():
            param.requires_grad = False
        for param in self.to_visual_latent.parameters():
            param.requires_grad = False
            
    def open_partial_vision_encoder(self):
        for name, param in self.visual_transformer.named_parameters():
            if 'to_pixels' in name or 'enc_temporal_transformer.layers' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        for param in self.to_visual_latent.parameters():
            param.requires_grad = True
            
    def open_vision_encoder(self):
        for param in self.visual_transformer.parameters():
            param.requires_grad = True
        for param in self.to_visual_latent.parameters():
            param.requires_grad = True
    
    def open_fusion_module(self):
        for param in self.fusion_module.parameters():
            param.requires_grad = True
            
    def freeze_fusion_module(self):
        for param in self.fusion_module.parameters():
            param.requires_grad = False
    def tokenize(self, prompt):
        text_tokens=self.tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(torch.cuda)
        return text_tokens
    
    def token_embedding(self,input_ids):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        if hasattr(self.text_transformer.embeddings, "token_type_ids"):
            print("hahatrue")

        buffered_token_type_ids = self.text_transformer.embeddings.token_type_ids[:, :seq_length]
        buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
        token_type_ids = buffered_token_type_ids_expanded
        text_embeddings = self.text_transformer.embeddings(input_ids = input_ids, token_type_ids = token_type_ids)
        return text_embeddings
    
    def multi_positive_infoNCE_loss(self, pred_sim, true_sim, positive_threshold=0.7):
        """
        infoNCE改进版本，分母去掉positive的元素（除了对角线）
        
        参数：
            pred_sim (torch.Tensor): 预测的相似度矩阵，形状为(b, n, m)，b为batchsize，n为query数量，m为每个query的样本数。
            true_sim (torch.Tensor): 真实的相似度矩阵，形状与pred_sim相同。
            positive_threshold (float): 定义正样本
        """
        true_sim[true_sim>=positive_threshold] = 1
        true_sim[true_sim<positive_threshold] = 0
        
        diag_mask = torch.eye(true_sim.shape[0]).to(true_sim.device)
        diag_mask = repeat(diag_mask, 'n m -> b n m', b=true_sim.shape[0])
        undiag_mask = torch.zeros_like(true_sim)
        undiag_mask[true_sim==0] = 1
        mask = undiag_mask + diag_mask  # b n m

        logits_sum = torch.exp(pred_sim).mul(mask).sum(2)   # b n
        logits_norm = torch.exp(torch.diagonal(pred_sim, dim1=1, dim2=2)) / logits_sum  # b n
        loss = -1 * torch.log(logits_norm)
        loss = loss.mean(dim=1) # b 保留了每个anatomy的loss
        
        return loss
    
    def triplet_loss(self, pred_sim, true_sim, margin=0.2, gt_threshold=0.3, positive_threshold=0.7, negative_threshold=0.4, temp=0.07):
        """
        计算Triplet Loss，用于排序模型的训练。

        参数：
            pred_sim (torch.Tensor): 预测的相似度矩阵，形状为(b, n, m)，b为batchsize，n为query数量，m为每个query的样本数。
            true_sim (torch.Tensor): 真实的相似度矩阵，形状与pred_sim相同。
            margin (float): 正负样本之间的最小间隔。
            gt_threshold (float): 定义正负样本

        返回：
            torch.Tensor: 计算得到的Triplet Loss标量值。
        """
        # 确保输入张量的形状一致
        assert pred_sim.shape == true_sim.shape, "预测矩阵和真实矩阵形状不一致"
        b, n, m = pred_sim.shape
        # 生成三维掩码矩阵，标记所有满足 true_sim[i] > true_sim[j] 的位置
        # true_sim.unsqueeze(3) 形状为(b, n, m, 1), true_sim.unsqueeze(2) 形状为(b, n, 1, m)
        mask = (true_sim.unsqueeze(3) - true_sim.unsqueeze(2)) > gt_threshold  # 形状(n, m, m)
        abs_mask =(true_sim > positive_threshold).unsqueeze(3) # 同时满足ij>0.7
        mask = mask & abs_mask
        abs_mask =(true_sim < negative_threshold).unsqueeze(2) # 同时满足ik<0.4
        mask = mask & abs_mask
        # 计算预测相似度的差值矩阵 pred_sim[j] - pred_sim[i]
        diff = - pred_sim.unsqueeze(3) + pred_sim.unsqueeze(2)  # 形状(b, n, m, m)  # 逼近-margion # pred_sim.unsqueeze(3)-pred_sim.unsqueeze(2)涨到0.2
        magnify = true_sim.unsqueeze(3) - true_sim.unsqueeze(2)  # 形状(b, n, m, m)
        # 计算Triplet Loss各项：max(0, diff + margin)
        losses = torch.clamp_min(diff + margin, 1e-8)
        # softplus : log(1+exp(.../temp))
        losses = torch.log(1 + torch.exp((losses * magnify) / temp))    # true_sim.unsqueeze(3)-true_sim.unsqueeze(2)越大，weight越大
        # 应用掩码，仅保留有效正负对
        masked_losses = losses * mask.float()
        # 计算每个query的总损失和有效对数量
        sum_per_query = masked_losses.sum(dim=(2, 3))  # 形状(b, n)
        num_pairs_per_query = mask.sum(dim=(2, 3))     # 形状(b, n)
        # 处理无有效对的情况，避免除零
        avg_loss_per_query = sum_per_query / (num_pairs_per_query + 1e-10)   # 避免分母为0（全部满足）
        avg_loss_per_query[num_pairs_per_query == 0] = 0  # 无有效对的query损失置零（好像是多余的）
        # 返回batch平均损失
        valid_mask = num_pairs_per_query != 0
        valid_count = valid_mask.float().sum(dim=1) # b # n个sample中有几个loss不为0
        # avg_valid_loss = torch.where(
        #     valid_count > 0,
        #     (avg_loss_per_query * valid_mask.float()).sum(dim=1) / valid_count,
        #     torch.zeros_like(valid_count)
        # )
        # 7.28修改avg_valid_loss的计算方式，避免除0
        epsilon = 1e-8
        safe_valid_count = valid_count + epsilon  # 防止0
        avg_valid_loss = ((avg_loss_per_query * valid_mask.float()).sum(dim=1) / safe_valid_count) * (valid_count > 0)

        return avg_valid_loss, torch.sum(mask).item()

    def multi_positive_infoNCE_loss_uncon(self, pred_sim, true_sim, positive_threshold=0.7):
        """
        infoNCE改进版本，分母去掉positive的元素（除了对角线）
        """
        true_sim[true_sim>=positive_threshold] = 1
        true_sim[true_sim<positive_threshold] = 0
        
        diag_mask = torch.eye(true_sim.shape[0]).to(true_sim.device)
        undiag_mask = torch.zeros_like(true_sim)
        undiag_mask[true_sim==0] = 1
        mask = undiag_mask + diag_mask

        logits_sum = torch.exp(pred_sim).mul(mask).sum(1)
        logits_norm = torch.exp(torch.diag(pred_sim)) / logits_sum
        loss = -1 * torch.log(logits_norm)
        loss = loss.mean()
        
        return loss
    
    def triplet_loss_uncon(self, pred_sim, true_sim, margin=0.2, gt_threshold=0.3, positive_threshold=0.7, negative_threshold=0.4, temp=0.07):
        """
        计算Triplet Loss，用于排序模型的训练。

        参数：
            pred_sim (torch.Tensor): 预测的相似度矩阵，形状为(n, m)，n为query数量，m为每个query的样本数。
            true_sim (torch.Tensor): 真实的相似度矩阵，形状与pred_sim相同。
            margin (float): 正负样本之间的最小间隔。
            gt_threshold (float): 定义正负样本

        返回：
            torch.Tensor: 计算得到的Triplet Loss标量值。
        """
        # 确保输入张量的形状一致
        assert pred_sim.shape == true_sim.shape, "预测矩阵和真实矩阵形状不一致"
        n, m = pred_sim.shape
        # 生成三维掩码矩阵，标记所有满足 true_sim[i]-true_sim[j] > true_sim[i] - true_sim[k] 的位置
        # true_sim.unsqueeze(2) 形状为(n, m, 1), true_sim.unsqueeze(1) 形状为(n, 1, m)
        mask = (true_sim.unsqueeze(2) - true_sim.unsqueeze(1)) > gt_threshold  # 形状(n, m, m)
        abs_mask =(true_sim > positive_threshold).unsqueeze(2) # 同时满足ij>0.7
        mask = mask & abs_mask
        abs_mask =(true_sim < negative_threshold).unsqueeze(1) # 同时满足ik<0.4
        mask = mask & abs_mask
        # 计算预测相似度的差值矩阵 pred_sim[j] - pred_sim[i]
        diff = - pred_sim.unsqueeze(2) + pred_sim.unsqueeze(1)  # 形状(n, m, m)
        # 计算Triplet Loss各项：max(0, diff + margin)
        losses = torch.clamp_min(diff + margin, 1e-8)
        # softplus : log(1+exp(.../temp))
        losses = F.softplus(losses / temp) # torch.log(1 + torch.exp(losses / temp))
        # 应用掩码，仅保留有效正负对
        masked_losses = losses * mask.float()
        # 计算每个query的总损失和有效对数量
        sum_per_query = masked_losses.sum(dim=(1, 2))  # 形状(n,)
        num_pairs_per_query = mask.sum(dim=(1, 2)) + 1e-14     # 形状(n,)   # 避免分母为0（全部满足）
        # 处理无有效对的情况，避免除零
        avg_loss_per_query = sum_per_query / num_pairs_per_query.clamp(min=1)
        avg_loss_per_query[num_pairs_per_query == 0] = 0  # 无有效对的query损失置零
        # 返回batch平均损失
        return avg_loss_per_query.mean()


    def forward(
            self,
            text,   # B
            image,  # B local_B 1 H W D
            device,
            gt_similarity_matrix=None,  # B local_B local_B
            return_loss = False,
            return_latents = False,
            is_condition = True
    ):
        global h_r, w_r, z_r
        if is_condition:
            B, device = text.input_ids.shape[0], device

            # derive text mask

            text_mask =text.attention_mask

            if return_loss:
                assert gt_similarity_matrix is not None, 'calculate loss but ground truth similarity matrix is not given'

            # get encoded text

            text_args = (text.input_ids,text.attention_mask)

            if not self.text_encode_without_mask:
                text_args = (*text_args, text_mask)

            text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
            enc_text = text_embeddings[0]   # B 512(token_len) 768
            
            # NOTE: need to flatten B local_B ahead
            B, local_B, _, D, H, W = image.shape
            image = rearrange(image, 'B N O D H W -> (B N) O D H W')
            # NOTE
            
            enc_image = self.visual_transformer(image, return_encoded_tokens=True)   # B*N 24 24 24 512(vis_dim)

            h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

            # NOTE: Do NOT Pooling. Instead, reshape to B (HWD) Dim

            image_embeds = rearrange(enc_image, 'b h w d dim -> h b (w d dim)')   # 24, B*N, (24*24*512)
            # 这里有问题，实际上是B T H W Dim，不过结果没有问题，就是 T B (H W Dim)

            # image_embeds = rearrange(enc_image, 'b h w d dim -> b (h w d) dim')   # B*N, 24*24*24, 512
            # NOTE

            # project to latents
            text_embeds = enc_text[:,0,:] # B 768

            text_latents = self.to_text_latent(text_embeds) # B 512

            image_latents = self.to_visual_latent(image_embeds) # 24 B*N 512

            # NOTE: Add a Cross-Attn here 
            # WARNING Batchsize on the 2nd dim, Squence_len on the 1st dim
            # text_latents, image_latents = map(l2norm, (text_latents, image_latents))
            text_latents = repeat(text_latents, 'b dim -> one (b n) dim', one=1, n=local_B) # 1 B*N 512

            t = image_latents.shape[0]
            temp_pos_embedding = self.pos_embedding[:,:t, :]  # 1 t 512

            pos = repeat(temp_pos_embedding, 'one d dim -> d (one bn) dim', bn=image_latents.shape[1]).to(device)  # 24, B*N, (24*24*512)
            fused_latents, _ = self.fusion_module(text_latents, image_latents, pos=pos) # 1 B*N 512

            
            fused_latents = fused_latents[0]
            
            norms = torch.norm(fused_latents, dim=-1)
            # print('fused_latents范数:', '最小值:', norms.min(), '最大值:', norms.max(), '存在零:', (norms == 0).any())
            if (norms < 1e-6).any():
                print('警告：检测到极小范数')
                # 可选：添加微小噪声避免零范数问题
                fused_latents = fused_latents + torch.randn_like(fused_latents) * 1e-8
            if torch.isnan(fused_latents).any() or torch.isinf(fused_latents).any():
                print('NaN/Inf in vec **before** l2norm')
        
            fused_latents = l2norm(fused_latents) # B*N 512
            
            
            if fused_latents.isnan().any() or fused_latents.isinf().any():
                print('NaN/Inf in vec **after** l2norm')
            if (fused_latents == 0).any():
                print('Zero in vec **after** l2norm')
            
            fused_latents = rearrange(fused_latents, '(b n) dim -> b n dim', n=local_B)
            # NOTE


            if return_latents:
                return text_latents, image_latents, fused_latents, self.temperature.exp()

            
            pred_similarity_matrix = einsum('b n d, k m d -> b k n m', (fused_latents, fused_latents))   # B B local_B local_B
            batch_indices = torch.arange(B)
            pred_similarity_matrix = pred_similarity_matrix[batch_indices, batch_indices, :, :] # B local_B local_B
            
            # early return, if needed

            if not return_loss:
                return pred_similarity_matrix

            # contrastive loss

            loss = 0
            
            if not (self.use_infoNCE_loss > 0 or self.use_triplet_loss > 0):
                raise ValueError("To Calculate Loss, use_triplet_loss and use_infoNCE_loss cannot both be False")
            
            if self.use_triplet_loss > 0:
                triplet_loss, valid_triplet_count = self.triplet_loss(pred_similarity_matrix, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
                # b
                loss += self.use_triplet_loss * triplet_loss.mean()
            else:
                triplet_loss = torch.zeros(B)
                valid_triplet_count = 0
                
            if self.use_infoNCE_loss > 0:
                infoNCE_loss = self.multi_positive_infoNCE_loss(pred_similarity_matrix, gt_similarity_matrix, positive_threshold=self.positive_threshold)
                # b
                loss += self.use_infoNCE_loss * infoNCE_loss.mean()
            else:
                infoNCE_loss = torch.zeros(B)

            # calculate CL loss

            return loss, triplet_loss, infoNCE_loss, valid_triplet_count
        else:
            b, device = text.input_ids.shape[0], device
                # derive text mask

            text_mask =text.attention_mask

            if return_loss:
                assert gt_similarity_matrix is not None, 'calculate loss but ground truth similarity matrix is not given'

            # concat augmented texts and images and do some asserts

            num_batch_texts = num_batch_images = 1
            text_args = (text.input_ids,text.attention_mask)

            if not self.text_encode_without_mask:
                text_args = (*text_args, text_mask)

            text_embeddings = self.text_transformer(text.input_ids, attention_mask = text.attention_mask )
            enc_text = text_embeddings[0]   # B 512(token_len) 768
            enc_image= self.visual_transformer(image, return_encoded_tokens=True)   # B 24 24 24 512(vis_dim)

    
            h_r, w_r, z_r = enc_image.shape[1], enc_image.shape[2], enc_image.shape[3]

            enc_image_send = enc_image

            enc_image = torch.mean(enc_image, dim=1)    # B 24 24 512 (avg over z-axis)

            enc_image = enc_image.view(enc_image.shape[0], -1)  # B 24*24*512
            # project to latents
            if self.use_all_token_embeds:
                assert enc_text.ndim == 3, 'encoded text must have 3 dimensions (batch, seq, features)'
                assert enc_image.ndim == 3, 'encoded image must have 3 dimensions (batch, seq [height x width], features)'
                text_embeds = enc_text[:, 1:] if self.text_has_cls_token else enc_text
                image_embeds = enc_image[:, 1:] if self.visual_has_cls_token else enc_image
            else:
                text_embeds = enc_text[:, :] if enc_text.ndim == 3 else enc_text # B 512(token_len) 768
                image_embeds = enc_image[:, :] if enc_image.ndim == 3 else enc_image # B 24*24*512

            text_embeds = text_embeds[:,0,:] # B 768

            text_latents = self.to_text_latent(text_embeds) # B 512

            image_latents = self.to_visual_latent(image_embeds) # B 512

            text_latents, image_latents = map(l2norm, (text_latents, image_latents))

            # whether to early return latents

            if return_latents:
                return text_latents, image_latents, enc_image_send, self.temperature.exp()

            # get temperature

            temp = self.temperature.exp()
            text_latents = rearrange(text_latents, '(m b) ... -> m b ...', m = num_batch_texts) # 1 B 512
            image_latents = rearrange(image_latents, '(m b) ... -> m b ...', m = num_batch_images) # 1 B 512
            
            text_to_image = einsum('m t d, n i d -> m n t i', text_latents, image_latents) * temp   # 1 1 B B
            image_to_text = rearrange(text_to_image, '... t i -> ... i t')
            
            if self.use_image2image_loss:
                image_to_image = einsum('m t d, n i d -> m n t i', image_latents, image_latents) * temp   # 1 1 B B
                image_to_image = image_to_image.squeeze()
                
            # calculate loss

            text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...').squeeze()    # 1 B B -> B B
            image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...').squeeze()    # 1 B B -> B B
            
            # NOTE: Official loss calculation only treat diagonal as positive samples (1), others as negative (0)
            # NOTE: Our implementation with soft similarity between each sample pair
            
            # gt_similarity_matrix = F.normalize(gt_similarity_matrix, dim=0) # B B
            
            if not (self.use_uncon_infoNCE_loss > 0 or self.use_uncon_triplet_loss > 0 or self.use_image2image_loss > 0):
                raise ValueError("To Calculate Loss, use_triplet_loss and use_infoNCE_loss and use_image2image_loss cannot all be False")
            
            text_to_image_triplet_loss = torch.zeros(1).to(device)
            image_to_text_triplet_loss = torch.zeros(1).to(device)
            if self.use_uncon_triplet_loss > 0:
                text_to_image_triplet_loss = self.triplet_loss_uncon(text_to_image, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
                image_to_text_triplet_loss = self.triplet_loss_uncon(image_to_text, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
            image_text_triplet_loss = (text_to_image_triplet_loss + image_to_text_triplet_loss) / 2
            
            image_to_image_triplet_loss = torch.zeros(1).to(device)
            if self.use_image2image_loss > 0:
                image_to_image_triplet_loss = self.triplet_loss_uncon(image_to_image, gt_similarity_matrix, margin=self.triplet_loss_margin, gt_threshold=self.positive_distance_threshold, positive_threshold=self.positive_threshold, negative_threshold=self.negative_threshold)
            
            text_to_image_infoNCE_loss = torch.zeros(1).to(device)
            image_to_text_infoNCE_loss = torch.zeros(1).to(device)
            if self.use_uncon_infoNCE_loss > 0:
                text_to_image_infoNCE_loss = self.multi_positive_infoNCE_loss_uncon(text_to_image, gt_similarity_matrix, positive_threshold=self.positive_threshold)
                image_to_text_infoNCE_loss = self.multi_positive_infoNCE_loss_uncon(image_to_text, gt_similarity_matrix, positive_threshold=self.positive_threshold)
            image_text_infoNCE_loss = (text_to_image_infoNCE_loss + image_to_text_infoNCE_loss) / 2

            # calculate CL loss
            
            cl_losses = torch.zeros(1).to(device)
            # cl_losses += self.use_triplet_loss * image_text_triplet_loss
            cl_losses += self.use_image2image_loss * image_to_image_triplet_loss
            cl_losses += self.use_uncon_infoNCE_loss * image_text_infoNCE_loss

            return cl_losses, image_text_triplet_loss, image_text_infoNCE_loss, image_to_image_triplet_loss

if __name__ == '__main__':
    
    from transformer_maskgit import CTViT
    
    device = torch.device('cuda:0')
    
    tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
    text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")


    image_encoder = CTViT(
        dim = 512,
        codebook_size = 8192,
        image_size = 480,
        patch_size = 20,
        temporal_patch_size = 10,
        spatial_depth = 4,
        temporal_depth = 4,
        dim_head = 32,
        heads = 8
    )

    clip = CTCLIP(
        tokenizer=tokenizer,
        image_encoder = image_encoder,
        text_encoder = text_encoder,
        dim_text = 768,
        dim_image = 294912,
        dim_latent = 512,
        extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
        use_mlm=False,
        downsample_image_embeds = False,
        use_all_token_embeds = False
    ).to(device)
    
    gt_similarity_matrix = torch.eye(3).to(device)
    gt_similarity_matrix[0,2] = 1
    gt_similarity_matrix[2,0] = 1
    print('Condition 3D')
    video = torch.ones(size=(3, 1 , 1 , 240, 480, 480)).to(device) # 依次表示batchsize，channels,D ,H , W
    
    
    text = ['text1 and text 1', 'text2','text3']
    text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    
    # loss = clip(text_tokens, video, gt_similarity_matrix, gt_similarity_matrix=gt_similarity_matrix, return_loss=True, device=device)
    text_latents, image_latents, enc_image_send,_ = clip(text_tokens, video, return_latents=True, device=device)
    print(text_latents.shape, image_latents.shape, enc_image_send.shape)
    # loss, triplet_loss, infoNCE_loss, valid_triplet_count = clip(text_tokens, video, gt_similarity_matrix =  gt_similarity_matrix, return_loss=True, device=device)
    # print('Loss:', loss.item(), 'Triplet Loss:', triplet_loss.item(), 'InfoNCE Loss:', infoNCE_loss.item(), 'Valid Triplet Count:', valid_triplet_count)


    print('Condition 3D')
    video = torch.ones(size=(3, 1 , 1 , 120, 480, 480)).to(device) # 依次表示batchsize，channels,D ,H , W

    text = ['text1 and text 1', 'text2','text3']
    text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    
    # loss = clip(text_tokens, video, gt_similarity_matrix, gt_similarity_matrix=gt_similarity_matrix, return_loss=True, device=device)
    text_latents, image_latents, enc_image_send,_ = clip(text_tokens, video, return_latents=True, device=device)
    print(text_latents.shape, image_latents.shape, enc_image_send.shape)
    # loss, triplet_loss, infoNCE_loss, valid_triplet_count = clip(text_tokens, video, gt_similarity_matrix = gt_similarity_matrix, return_loss=True, device=device)
    # print('Loss:', loss.item(), 'Triplet Loss:', triplet_loss.item(), 'InfoNCE Loss:', infoNCE_loss.item(), 'Valid Triplet Count:', valid_triplet_count)

    print('Condition 2D')
    video = torch.ones(size=(3, 1, 1, 1, 480, 480)).to(device)

    text = ['text1 and text 1', 'text2','text3']
    text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

    # loss = clip(text_tokens, video, gt_similarity_matrix, gt_similarity_matrix=gt_similarity_matrix, return_loss=True, device=device)
    text_latents, image_latents, enc_image_send,_ = clip(text_tokens, video, return_latents=True, device=device)
    print(text_latents.shape, image_latents.shape, enc_image_send.shape)
    
    # loss, triplet_loss, infoNCE_loss, valid_triplet_count = clip(text_tokens, video, gt_similarity_matrix = gt_similarity_matrix, return_loss=True, device=device)
    # print('Loss:', loss.item(), 'Triplet Loss:', triplet_loss.item(), 'InfoNCE Loss:', infoNCE_loss.item(), 'Valid Triplet Count:', valid_triplet_count)

    print('Uncondition 3D')
    video = torch.ones(size=(3,  1 , 240, 480, 480)).to(device) # 依次表示batchsize，channels,D ,H , W

    text = ['text1 and text 1', 'text2','text3']
    text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    
    text_latents, image_latents, enc_image_send,_ = clip(text_tokens, video, return_latents=True, device=device,is_condition=False)
    print(text_latents.shape, image_latents.shape, enc_image_send.shape)

    cl_losses, image_text_triplet_loss, image_text_infoNCE_loss, image_to_image_triplet_loss = clip(text_tokens, video, gt_similarity_matrix = gt_similarity_matrix, return_loss=True, device=device, is_condition=False)
    print('CL Loss:', cl_losses.item(), 'Image-Text Triplet Loss:', image_text_triplet_loss.item(), 'Image-Text InfoNCE Loss:', image_text_infoNCE_loss.item(), 'Image-to-Image Triplet Loss:', image_to_image_triplet_loss.item())

    print('Uncondition 3D')
    video = torch.ones(size=(3,  1 , 120, 480, 480)).to(device) # 依次表示batchsize，channels,D ,H , W

    text = ['text1 and text 1', 'text2','text3']
    text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

    text_latents, image_latents, enc_image_send,_ = clip(text_tokens, video, return_latents=True, device=device,is_condition=False)
    print(text_latents.shape, image_latents.shape, enc_image_send.shape)

    cl_losses, image_text_triplet_loss, image_text_infoNCE_loss, image_to_image_triplet_loss = clip(text_tokens, video, gt_similarity_matrix = gt_similarity_matrix, return_loss=True, device=device, is_condition=False)
    print('CL Loss:', cl_losses.item(), 'Image-Text Triplet Loss:', image_text_triplet_loss.item(), 'Image-Text InfoNCE Loss:', image_text_infoNCE_loss.item(), 'Image-to-Image Triplet Loss:', image_to_image_triplet_loss.item())


    print('Uncondition 2D')
    video = torch.ones(size=(3,  1 , 1, 480, 480)).to(device) # 依次表示batchsize，channels,D ,H , W

    text = ['text1 and text 1', 'text2','text3']
    text_tokens = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)

    text_latents, image_latents, enc_image_send,_ = clip(text_tokens, video, return_latents=True, device=device,is_condition=False)
    print(text_latents.shape, image_latents.shape, enc_image_send.shape)

    cl_losses, image_text_triplet_loss, image_text_infoNCE_loss, image_to_image_triplet_loss = clip(text_tokens, video, gt_similarity_matrix = gt_similarity_matrix, return_loss=True, device=device, is_condition=False)
    print('CL Loss:', cl_losses.item(), 'Image-Text Triplet Loss:', image_text_triplet_loss.item(), 'Image-Text InfoNCE Loss:', image_text_infoNCE_loss.item(), 'Image-to-Image Triplet Loss:', image_to_image_triplet_loss.item())
    