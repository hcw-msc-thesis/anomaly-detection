from typing import Any

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from anomalib import LearningType
from anomalib.data import InferenceBatch
from anomalib.models.components import AnomalibModule, KCenterGreedy, MemoryBankMixin
from torch import nn


class FPFHFeatures(nn.Module):
    """Custom Pytorch FPFH model for use in Anomalib"""
    def __init__(self, 
            patch =  4,
            coreset_sampling_ratio = 0.1,
            normal_nn = 30,
            feature_nn = 100,
            scales = None,
            min_coverage = 0.5
        ):
        if scales is None:
            scales = [1, 3, 6]
        super().__init__()
        self.patch = patch
        self.coreset_sampling_ratio=coreset_sampling_ratio
        self.normal_nn=normal_nn
        self.feature_nn=feature_nn 
        self.scales=scales
        self.min_coverage=min_coverage
        self.register_buffer("memory_bank", torch.empty(0))
        self.register_buffer("block_spreads", torch.ones(len(scales)))

    @torch.no_grad()
    def extract_single_map(self, depth_tensor) -> torch.Tensor:
        """Get FPFH features from a sample"""
        depth = depth_tensor.permute(1, 2, 0).detach().cpu().numpy()
        h, w, _ = depth.shape
        points = depth.reshape(-1, 3)

        #Remove background points
        valid_mask = points[:, 2] != 0
        valid_points = points[valid_mask]

        #If given a blank image
        if len(valid_points) == 0:
            grid_h = h // self.patch
            grid_w = w // self.patch
            return torch.zeros((33 * len(self.scales), grid_h, grid_w), device=depth_tensor.device), torch.zeros((1, grid_h, grid_w), device=depth_tensor.device) 

        #Reshape point cloud data
        object_map = (torch.from_numpy(valid_mask.reshape(h, w))
            .to(depth_tensor.device, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        #Get features from reshaped point cloud
        pools = []
        pooled_object = F.avg_pool2d(object_map, kernel_size=self.patch, stride=self.patch).squeeze(0)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(valid_points)
        #Per RIAD, multiple scale masks
        for scale in self.scales:
            point_cloud.estimate_normals(
                o3d.geometry.KDTreeSearchParamKNN(knn=scale*self.normal_nn)
            )
            #Ensure normals face camera
            point_cloud.orient_normals_towards_camera_location(np.zeros(3))
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                point_cloud,
                o3d.geometry.KDTreeSearchParamKNN(knn=scale*self.feature_nn),
            )

            full_fpfh = np.zeros((h * w, 33), dtype=np.float32)
            features = np.asarray(fpfh.data).T
            full_fpfh[valid_mask] = (features / np.maximum(features.sum(1, keepdims=True), 1e-8))

            fpfh_map = (
                torch.from_numpy(full_fpfh.reshape(h, w, 33))
                .to(depth_tensor.device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )

            pooled_features = F.avg_pool2d(fpfh_map, kernel_size=self.patch, stride=self.patch).squeeze(0)
            object_only_average = pooled_features / pooled_object.clamp(min=1e-4)
            pools.append(object_only_average)

        concatenated = torch.cat(pools, dim=0)
        return concatenated, pooled_object

    def forward(self, depth_map) -> InferenceBatch | torch.Tensor:
        _, _, h, w = depth_map.shape
        batch_patches = []
        batch_coverage = []

        for depth in depth_map:
            fmap, coverage = self.extract_single_map(depth)
            coverage_value = coverage.reshape(-1)
            patches = fmap.permute(1, 2, 0).reshape(-1, fmap.shape[0])
            batch_patches.append(patches)
            batch_coverage.append(coverage_value)

        if self.training:
            kept_patches = [patch[coverage >= self.min_coverage] for patch, coverage in zip(batch_patches, batch_coverage)]
            return torch.vstack(kept_patches)

        anomaly_maps = []
        gh, gw = fmap.shape[1:]
        for patches, coverage in zip(batch_patches, batch_coverage):
            keep = coverage >= self.min_coverage
            patches = patches[keep] / self.block_spreads.repeat_interleave(33)
            cell_dists = torch.zeros(gh * gw, device=patches.device)
            dists = torch.cdist(patches, self.memory_bank)
            min_dists, _ = torch.min(dists, dim=1)
            cell_dists[keep] = min_dists
            amap = cell_dists.reshape(1, 1, gh, gw)
            amap = F.interpolate(amap, size=(h, w), mode="bilinear", align_corners=False)
            anomaly_maps.append(amap.squeeze(0))

        anomaly_map = torch.stack(anomaly_maps)
        pred_score = anomaly_map.flatten(1).max(dim=1).values
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)

class FPFHAnomalib(MemoryBankMixin, AnomalibModule):

    def __init__(
        self,
        patch = 4,
        coreset_sampling_ratio = 0.1,
        normal_nn = 30,
        feature_nn = 100,
        scales = None,
        min_coverage = 0.5
    ):
        if scales is None:
            scales = [1, 3, 6]
        super().__init__()
        self.save_hyperparameters()
        self.model = FPFHFeatures(
            patch=patch, 
            normal_nn=normal_nn, 
            feature_nn=feature_nn, 
            scales=scales, 
            min_coverage=min_coverage
        )
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.embeddings: list[torch.Tensor] = []
        self.pre_processor = None  # Keep raw 3D XYZ coordinates untouched

    @property
    def learning_type(self) -> LearningType:
        return LearningType.ONE_CLASS

    @property
    def trainer_arguments(self) -> dict[str, Any]:
        return {"max_epochs": 1, "gradient_clip_val": 0}

    def configure_optimizers(self):
        return None 

    def training_step(self, batch, *args, **kwargs):
        embeddings = self.model(batch.depth_map.cpu()).to(self.device)
        self.embeddings.append(embeddings)
        return None

    def fit(self):
        rng = np.random.default_rng(0)
        block_spreads = []
        all_embeddings = torch.vstack(self.embeddings)
        for i, scale in enumerate(self.model.scales):
            block = all_embeddings[:, 33*i : 33*(i+1)]
            index_a, index_b = rng.integers(block.shape[0], size=(2, 20000))
            diff = block[index_a] - block[index_b]
            distances = torch.linalg.norm(diff, dim=1)
            median = distances.median()
            block_spreads.append(median)

        self.model.block_spreads = torch.stack(block_spreads)
        all_embeddings = all_embeddings / self.model.block_spreads.repeat_interleave(33)
        sampler = KCenterGreedy(
            embedding=all_embeddings,
            sampling_ratio=self.coreset_sampling_ratio,
        )
        self.model.memory_bank = sampler.sample_coreset()
        self.embeddings.clear()

    def validation_step(self, batch, *args, **kwargs):
        predictions = self.model(batch.depth_map)
        return batch.update(**predictions._asdict())
