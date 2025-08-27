import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..ops import multi_head_attention_dispatch

torch.backends.cudnn.deterministic = True


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int = 2, descriptor_dim: int = 256, num_heads: int = 4, gamma: float = 1.0) -> None:
        print(f"LearnableFourierPositionalEncoding: M={M}, descriptor_dim={descriptor_dim}, num_heads={num_heads}, gamma={gamma}")
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = descriptor_dim // num_heads
        self.Wr = nn.Linear(M, self.head_dim // 2, bias=False)
        self.gamma = gamma
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.sin(projected + torch.pi/2), torch.sin(projected)
        emb = torch.stack([cosines, sines])
        return emb.repeat_interleave(2, dim=3).repeat(1, 1, 1, self.num_heads).unsqueeze(4)

class LearnableFourierPositionalEncodingTIDL(LearnableFourierPositionalEncoding):
    def __init__(self, M: int = 2, descriptor_dim: int = 256, num_heads: int = 4, gamma: float = 1.0) -> None:
        super().__init__(M, descriptor_dim, num_heads, gamma)
        self.interleaved_indices = [i for i in range(self.head_dim // 2) for _ in range(2)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #TODO: fix hardcoded sizes
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.sin(projected + torch.pi/2), torch.sin(projected)
        emb = torch.stack([cosines, sines])
        emb = emb.reshape((2 * 4096, 32))
        interleaved_emb = emb[:, self.interleaved_indices]
        repeated = torch.cat([interleaved_emb] * self.num_heads, dim=-1)
        repeated = repeated.reshape(2, 4096, 256)
        return repeated.unsqueeze(-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

def multi_head_attention_tidl(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    n, d = q.shape
    head_dim = d // num_heads
    q, k, v = (t.reshape((n, num_heads, head_dim)).transpose(0, 1) for t in (q, k, v))
    return F.scaled_dot_product_attention(q, k, v).transpose(0, 1).reshape((n, d))

class SelfBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        print(f"SelfBlock: embed_dim={embed_dim}, num_heads={num_heads}, bias={bias}")
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv: torch.Tensor = self.Wqkv(x)
        qkv = qkv.reshape((b, n, self.embed_dim, 3))
        qk, v = qkv[..., :2], qkv[..., 2]
        qk = self.apply_cached_rotary_emb(encoding, qk)
        q, k = qk[..., 0], qk[..., 1]
        context = multi_head_attention_dispatch(q, k, v, self.num_heads)
        message = self.out_proj(context)
        return x + self.ffn(torch.concat([x, message], 2))

    def rotate_half(self, qk: torch.Tensor) -> torch.Tensor:
        b, n, _, _ = qk.shape
        qk = qk.reshape((b, n, self.num_heads, self.head_dim // 2, 2, 2))
        qk = torch.stack((-qk[..., 1, :], qk[..., 0, :]), dim=4)
        qk = qk.reshape((b, n, self.embed_dim, 2))
        return qk

    def apply_cached_rotary_emb(self, encoding: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
        return qk * encoding[0] + self.rotate_half(qk) * encoding[1]

class SelfBlockTIDL(SelfBlock):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, bias)

    def rotate_half(self, qk: torch.Tensor) -> torch.Tensor:
        n, _, _ = qk.shape
        qk = qk.reshape((n, self.num_heads * self.head_dim // 2, 2, 2))
        qk = torch.stack((-qk[..., 1, :], qk[..., 0, :]), dim=-2)
        qk = qk.reshape((n, self.embed_dim, 2))
        return qk

    def apply_cached_rotary_emb(self, encoding: torch.Tensor, qk: torch.Tensor) -> torch.Tensor:
        encoding0, encoding1 = encoding.split(1, dim=0)
        encoding0 = encoding0.squeeze(0)
        encoding1 = encoding1.squeeze(0)
        return qk * encoding0 + self.rotate_half(qk) * encoding1

    def forward(self, x: torch.Tensor, encoding: torch.Tensor) -> torch.Tensor:
        n, _ = x.shape
        qkv: torch.Tensor = self.Wqkv(x)
        qkv = qkv.reshape((n, self.embed_dim, 3))
        qk, v = qkv[..., :2], qkv[..., 2]
        qk = self.apply_cached_rotary_emb(encoding, qk)
        q, k = qk[..., 0], qk[..., 1]
        context = multi_head_attention_tidl(q, k, v, self.num_heads)
        message = self.out_proj(context)
        return x + self.ffn(torch.concat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        print(f"CrossBlock: embed_dim={embed_dim}, num_heads={num_heads}, bias={bias}")
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.to_qk = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        b, _, _ = descriptors.shape
        qk, v = self.to_qk(descriptors), self.to_v(descriptors)

        indices = torch.arange(b, device=descriptors.device)
        swap = (indices // 2) * 2 + (1 - indices % 2)  # swap trick
        m = multi_head_attention_dispatch(qk, qk[swap], v[swap], self.num_heads)
        m = self.to_out(m)
        descriptors = descriptors + self.ffn(torch.concat([descriptors, m], 2))
        return descriptors

class CrossBlockTIDL(CrossBlock):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
        super().__init__(embed_dim, num_heads, bias)

    def forward(self, descriptors0: torch.Tensor, descriptors1: torch.Tensor) -> torch.Tensor:
        #TODO: calculate cross attention in both ways in one go
        qk0 = self.to_qk(descriptors0)
        qk1, v1 = self.to_qk(descriptors1), self.to_v(descriptors1)
        m = multi_head_attention_tidl(qk0, qk1, v1, self.num_heads)
        m = self.to_out(m)
        descriptors0 = descriptors0 + self.ffn(torch.concat([descriptors0, m], -1))
        return descriptors0


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads)
        self.cross_attn = CrossBlock(embed_dim, num_heads)

    def forward(self, descriptors: torch.Tensor, encodings: torch.Tensor) -> torch.Tensor:
        descriptors = self.self_attn(descriptors, encodings)
        return self.cross_attn(descriptors)

class TransformerLayerTIDL(TransformerLayer):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__(embed_dim, num_heads)
        self.self_attn = SelfBlockTIDL(embed_dim, num_heads)
        self.cross_attn = CrossBlockTIDL(embed_dim, num_heads)

    def forward(self, descriptors0: torch.Tensor, descriptors1: torch.Tensor, encodings0: torch.Tensor, encodings1: torch.Tensor) -> torch.Tensor:
        descriptors0 = self.self_attn(descriptors0, encodings0)
        descriptors1 = self.self_attn(descriptors1, encodings1)
        m0 = self.cross_attn(descriptors0, descriptors1)
        m1 = self.cross_attn(descriptors1, descriptors0)
        return m0, m1



def sigmoid_log_double_softmax(similarities: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    certainties = F.logsigmoid(z[0::2]) + F.logsigmoid(z[1::2]).transpose(1, 2)
    scores0 = F.log_softmax(similarities, 2)
    scores1 = F.log_softmax(similarities, 1)
    scores = scores0 + scores1 + certainties
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = dim**0.25
        self.final_proj = nn.Linear(dim, dim, bias=True)
        self.matchability = nn.Linear(dim, 1, bias=True)

    def forward(self, descriptors: torch.Tensor) -> torch.Tensor:
        """build assignment matrix from descriptors"""
        mdescriptors = self.final_proj(descriptors) / self.scale
        similarities = mdescriptors[0::2] @ mdescriptors[1::2].transpose(1, 2)
        z = self.matchability(descriptors)
        scores = sigmoid_log_double_softmax(similarities, z)
        return scores

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, threshold: float):
    """obtain matches from a log assignment matrix [BxNxN]"""
    max0 = torch.topk(scores, k=1, dim=2, sorted=False)  # scores.max(2)
    max1 = torch.topk(scores, k=1, dim=1, sorted=False)  # scores.max(1)
    m0, m1 = max0.indices[:, :, 0], max1.indices[:, 0, :]

    indices = torch.arange(m0.shape[1], device=m0.device).expand_as(m0)
    mutual = indices == m1.gather(1, m0)
    mscores = max0.values[:, :, 0].exp()
    valid = mscores > threshold

    b_idx, m0_idx = torch.where(valid & mutual)
    m1_idx = m0[b_idx, m0_idx]
    matches = torch.concat([b_idx[:, None], m0_idx[:, None], m1_idx[:, None]], 1)
    mscores = mscores[b_idx, m0_idx]
    return matches, mscores


class LightGlue(nn.Module):
    def __init__(
        self,
        url: str,
        input_dim: int = 256,
        descriptor_dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 9,
        filter_threshold: float = 0.1,  # match threshold
        depth_confidence: float = -1,  # -1 is no early stopping, recommend: 0.95
        width_confidence: float = -1,  # -1 is no point pruning, recommend: 0.99
    ) -> None:
        super().__init__()

        self.descriptor_dim = descriptor_dim
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.filter_threshold = filter_threshold
        self.depth_confidence = depth_confidence
        self.width_confidence = width_confidence

        if input_dim != self.descriptor_dim:
            self.input_proj = nn.Linear(input_dim, self.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        self.posenc = LearnableFourierPositionalEncoding(2, self.descriptor_dim, self.num_heads)

        d, h, n = self.descriptor_dim, self.num_heads, self.n_layers

        self.transformers = nn.ModuleList([TransformerLayer(d, h) for _ in range(n)])

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])
        self.register_buffer(
            "confidence_thresholds",
            torch.Tensor([self.confidence_threshold(i) for i in range(n)]),
        )

        state_dict = torch.hub.load_state_dict_from_url(url)

        # rename old state dict entries
        for i in range(n):
            pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
            state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
        self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        keypoints: torch.Tensor,  # (2B, N, 2), normalized
        descriptors: torch.Tensor,  # (2B, N, D)
    ):
        descriptors = self.input_proj(descriptors)

        # positional embeddings
        encodings = self.posenc(keypoints)  # (2, 2B, *, 64, 1)

        # GNN + final_proj + assignment
        for i in range(self.n_layers):
            # self+cross attention
            descriptors = self.transformers[i](descriptors, encodings)

        scores = self.log_assignment[i](descriptors)  # (B, N, N)
        matches, mscores = filter_matches(scores, self.filter_threshold)
        return matches, mscores  # (M, 3), (M,)

    def confidence_threshold(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self,
        confidences: torch.Tensor | None,
        scores: torch.Tensor,
        layer_index: int,
    ) -> torch.Tensor:
        """mask points which should be removed"""
        keep = scores > (1 - self.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            keep |= confidences <= self.confidence_thresholds[layer_index]
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.depth_confidence

class LightGlueTIDL(LightGlue):
    def __init__(
        self,
        url: str,
        input_dim: int = 256,
        descriptor_dim: int = 256,
        num_heads: int = 4,
        n_layers: int = 9,
        filter_threshold: float = 0.1,
        depth_confidence: float = -1,
        width_confidence: float = -1,
    ) -> None:
        super().__init__(url, input_dim, descriptor_dim, num_heads, n_layers, filter_threshold, depth_confidence, width_confidence)

        if input_dim != self.descriptor_dim:
            self.input_proj = nn.Linear(input_dim, self.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        # self.posenc = SimlifiedPositionalEncoding(2, self.descriptor_dim, self.num_heads)
        self.posenc = LearnableFourierPositionalEncodingTIDL(2, self.descriptor_dim, self.num_heads)

        d, h, n = self.descriptor_dim, self.num_heads, self.n_layers

        self.transformers = nn.ModuleList([TransformerLayerTIDL(d, h) for _ in range(n)])

    def forward(self, keypoints0: torch.Tensor, keypoints1: torch.Tensor, descriptors0: torch.Tensor, descriptors1: torch.Tensor):
        descriptors0 = self.input_proj(descriptors0)
        descriptors1 = self.input_proj(descriptors1)

        encodings0 = self.posenc(keypoints0)
        encodings1 = self.posenc(keypoints1)

        for i in range(self.n_layers):
            descriptors0, descriptors1 = self.transformers[i](descriptors0, descriptors1, encodings0, encodings1)
        

        return descriptors0, descriptors1
