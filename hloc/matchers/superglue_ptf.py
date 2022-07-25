from torch import nn
from copy import deepcopy
import torch
import numpy as np
from typing import Tuple
from keypoints.detect_point import detect_points

from ..utils.base_model import BaseModel


def attention(query, key, value):
    channels = query.shape[1]
    results = torch.einsum("abcd,abce->acde", query, key) / channels**0.5
    soft_out = torch.nn.functional.softmax(results, dim=-1)
    return torch.einsum("acde,abce->abcd", soft_out, value), soft_out


def points_normalization(points, shape, scale_factor=0.7):
    """
    нормализация точек
    """
    height, width = shape[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = torch.Tensor([width, height])[None].to(device)
    scale = (size.max() * scale_factor)[None][None]
    return (points - (size / 2)[None]) / scale[None]


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count: int, dim: int):
        super().__init__()
        assert dim % head_count == 0
        self.head_count = head_count
        self.dim = dim // head_count

        self.concat = nn.Conv1d(dim, dim, kernel_size=1)
        self.layers = nn.ModuleList([deepcopy(self.concat) for i in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [
            y(x).view(batch_dim, self.dim, self.head_count, -1)
            for y, x in zip(self.layers, (query, key, value))
        ]
        x, prob = attention(query, key, value)
        self.prob.append(prob)
        return self.concat(
            x.contiguous().view(batch_dim, self.dim * self.head_count, -1)
        )


def optimal_path(results, bin_sc, iters: int):

    dims, height, width = results.shape
    one = results.new_tensor(1)
    height_tensor, width_tensor = (height * one).to(results), (width * one).to(results)

    path0 = bin_sc.expand(dims, height, 1)
    path1 = bin_sc.expand(dims, 1, width)
    bin_sc = bin_sc.expand(dims, 1, 1)

    connects = torch.cat(
        [torch.cat([results, path0], -1), torch.cat([path1, bin_sc], -1)], 1
    )

    log_norm = -(height_tensor + width_tensor).log()
    log_height = torch.cat(
        [log_norm.expand(height), width_tensor.log()[None] + log_norm]
    )
    log_width = torch.cat(
        [log_norm.expand(width), height_tensor.log()[None] + log_norm]
    )
    log_height, log_width = log_height[None].expand(dims, -1), log_width[None].expand(
        dims, -1
    )

    u, v = torch.zeros_like(log_height), torch.zeros_like(log_width)
    for i in range(iters):
        u = log_height - torch.logsumexp(connects + v.unsqueeze(1), dim=2)
        v = log_width - torch.logsumexp(connects + u.unsqueeze(2), dim=1)
    connects = connects + u.unsqueeze(2) + v.unsqueeze(1)

    connects = connects - log_norm
    return connects


class Propagation(nn.Module):
    def __init__(self, dim: int, head_count: int):
        super().__init__()
        self.attn = MultiHeadedAttention(head_count, dim)
        self.perceptron = perceptron_layer([2 * dim, 2 * dim, dim])
        nn.init.constant_(self.perceptron[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.perceptron(torch.cat([x, message], dim=1))


class GNN(nn.Module):
    def __init__(self, dim: int, names0: list):
        super().__init__()
        self.layers = nn.ModuleList([Propagation(dim, 4) for i in range(len(names0))])
        self.layer_names = names0

    def forward(self, points_data0, points_data1):
        for layer, lname in zip(self.layers, self.layer_names):
            layer.attn.prob = []
            if lname == "cross":
                src0, src1 = points_data1, points_data0
            else:
                src0, src1 = points_data0, points_data1
            diff0, diff1 = layer(points_data0, src0), layer(points_data1, src1)
            points_data0, points_data1 = (points_data0 + diff0), (points_data1 + diff1)
        return points_data0, points_data1


def perceptron_layer(channels: list, add_batch_norm=True):
    """
    простейший персептрон
    """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if add_batch_norm:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """
    простой енкодер
    """

    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = perceptron_layer([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, points, results):
        inputs = [points.transpose(1, 2), results.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, 1))


class graph_detector(nn.Module):

    default_config = {
        "dim": 256,
        "weights": "graph",
        "encoder_layers": [32, 64, 128, 256],
        "GNN_layers": ["self", "cross"] * 9,
        "sinkhorn": 20,
        "match_threshold": 0.7,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.point_encoder = Encoder(self.config["dim"], self.config["encoder_layers"])

        self.gnn = GNN(self.config["dim"], self.config["GNN_layers"])

        self.final_proj = nn.Conv1d(
            self.config["dim"], self.config["dim"], kernel_size=1, bias=True
        )

        self.bin_score = torch.nn.Parameter(torch.tensor(1.0))

        path = self.config["weights_path"]
        self.load_state_dict(torch.load(path))
        print("Loaded graph detector model")

    def forward(self, data):
        resize = tuple(data["image_size0"].cpu().numpy().astype(np.uint))
        points_data0, points_data1 = data["descriptors0"], data["descriptors1"]
        points0, points1 = data["keypoints0"], data["keypoints1"]

        if points0.shape[1] == 0 or points1.shape[1] == 0:
            shape0, shape1 = points0.shape[:-1], points1.shape[:-1]
            return {
                "matches0": points0.new_full(shape0, -1, dtype=torch.int),
                "matches1": points1.new_full(shape1, -1, dtype=torch.int),
                "matching_scores0": points0.new_zeros(shape0),
                "matching_scores1": points1.new_zeros(shape1),
            }
        points0 = points_normalization(points0, resize)
        points1 = points_normalization(points1, resize)

        points_data0 = points_data0 + self.point_encoder(points0, data["scores0"])
        points_data1 = points_data1 + self.point_encoder(points1, data["scores1"])

        points_data0, points_data1 = self.gnn(points_data0, points_data1)

        final_points_data0, final_points_data1 = self.final_proj(
            points_data0
        ), self.final_proj(points_data1)

        results = torch.einsum("abc,abd->acd", final_points_data0, final_points_data1)
        results = results / self.config["dim"] ** 0.5

        results = optimal_path(results, self.bin_score, iters=self.config["sinkhorn"])

        max0, max1 = results[:, :-1, :-1].max(2), results[:, :-1, :-1].max(1)
        point_indices0, point_indices1 = max0.indices, max1.indices

        mindex0 = (
            point_indices0.new_ones(point_indices0.shape[1]).cumsum(0) - 1
        ) == point_indices1.gather(1, point_indices0)
        mindex1 = (
            point_indices1.new_ones(point_indices1.shape[1]).cumsum(0) - 1
        ) == point_indices0.gather(1, point_indices1)

        zero = results.new_tensor(0)
        mresults0 = torch.where(mindex0, max0.values.exp(), zero)
        mresults1 = torch.where(mindex1, mresults0.gather(1, point_indices1), zero)
        valid0 = mindex0 & (mresults0 > self.config["match_threshold"])
        valid1 = mindex1 & valid0.gather(1, point_indices1)
        point_indices0 = torch.where(
            valid0, point_indices0, point_indices0.new_tensor(-1)
        )
        point_indices1 = torch.where(
            valid1, point_indices1, point_indices1.new_tensor(-1)
        )

        return {
            "matches0": point_indices0,
            "matches1": point_indices1,
            "matching_scores0": mresults0,
            "matching_scores1": mresults1,
        }


class SuperGluePTF(BaseModel):
    def _init(self, conf):
        self.net = graph_detector(conf)

    def _forward(self, data):
        return self.net(data)
