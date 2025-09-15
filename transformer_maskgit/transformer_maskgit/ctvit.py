from pathlib import Path
import copy
import math
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import grad as torch_grad
from torchvision import transforms as T, utils


import torchvision

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from transformer_maskgit.attention import Attention, Transformer, ContinuousPositionBias,TransformerCLS

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def l2norm(t):
    return F.normalize(t, dim = -1)

# ctvit - 3d ViT with factorized spatial and temporal attention

def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    device=torch.device('cuda')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth,
        temporal_depth,
        cls_depth,
        discr_base_dim = 16,
        dim_head = 64,
        heads = 8,
        channels = 1,
        use_vgg_and_gan = True,
        vgg = None,
        discr_attn_res_layers = (16,),
        use_hinge_loss = True,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )
        # 模态嵌入层，设定了10个模态
        self.modal_embeddings = nn.Embedding(10, dim)
        
        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = False,
        )
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)
        self.enc_cls_transformer = TransformerCLS(depth = cls_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)

        
    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    def copy_for_eval(self):
        device = next(self.parameters()).device
        device=torch.device('cuda')
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    #@remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    #@remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)


    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens,
        is_condition=False
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])
        # print('before rearrange',tokens.shape)
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        device=torch.device('cuda')
        # print('before enc_spatial',tokens.shape)
        attn_bias = self.spatial_rel_pos_bias(h, w, device = device)
        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        del attn_bias
        torch.cuda.empty_cache()
        # print('after enc_spatial',tokens.shape)
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)
 
        # encode - temporal
        if tokens.shape[1] > 1:
            # 深度不为1，是3D图像
            tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
            tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)
            tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)
                   
        tokens = torch.mean(tokens, dim = 1)
        tokens = rearrange(tokens, 'b h w d -> b (h w) d')
        if is_condition:
            return tokens
        video_shape = list(video_shape)  # 这里的修改是因为enc_cls_transformer的内部需要
        video_shape[1] = 1  # 修改第二个元素
        # 转换回元组（如果需要）
        video_shape = tuple(video_shape)
        tokens = self.enc_cls_transformer(tokens, video_shape = video_shape)
        tokens = tokens[:,0] # 返回b d的结果
        return tokens

    def forward(
        self,
        video,
        mask = None,
        return_encoded_tokens=True,
        modal_embedding = False,
        modal_indexs = None,
        is_condition = False
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4
        #print(video.shape)

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        device=torch.device('cuda')
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f
        if video.shape[2] > 1:
            tokens = self.to_patch_emb(video)
        # save height and width in
        else:
            tokens = self.to_patch_emb_first_frame(video)

        shape = tokens.shape
        *_, h, w, _ = shape

        # encode - spatial
        if modal_embedding:
            modality_emb = self.modal_embeddings(modal_indexs)
            modality_emb = modality_emb.view(-1, 1, 1, 1, modality_emb.size(-1))
            tokens = tokens + modality_emb
        tokens = self.encode(tokens,is_condition)


        return tokens
        
    
if __name__ == '__main__':
    
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image_encoder = image_encoder.to(device)

    print('3D')
    video = torch.zeros((2, 1, 240, 480, 480)).to(device)
    result = image_encoder(video,return_encoded_tokens=True)
    print(result.shape)
    print('2D')
    video = torch.zeros((2, 1, 1, 480, 480)).to(device)
    result = image_encoder(video,return_encoded_tokens=True)
    print(result.shape)
