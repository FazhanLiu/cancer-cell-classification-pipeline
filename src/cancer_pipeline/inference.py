"""Unlabelled patch inference, independent of dataset manifests and test labels."""
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from cancer_pipeline.artifacts import checkpoint_model, file_hash, load_decision, object_hash
from cancer_pipeline.data import build_transforms
from cancer_pipeline.metrics import validate_predictions
from cancer_pipeline.utils import get_device
from cancer_pipeline.xai import attention_region_summary, explain, overlay_heatmap, predict_prob


class PatchInference:
    """One cached ViT; serialize access because rollout temporarily installs hooks."""

    def __init__(self, checkpoint_path, decision_path=None, device='auto', xai_options=None):
        self.device = get_device(device)
        self.model, self.config, checkpoint = checkpoint_model(checkpoint_path, self.device)
        self.provenance = checkpoint.get('provenance', {})
        if self.provenance.get('source_status', 'verified') != 'verified':
            raise ValueError('请使用来源已验证的训练权重，不能用历史来源未知的模型生成病例报告。')
        if not hasattr(self.model, 'blocks') or getattr(self.model, 'num_prefix_tokens', 1) != 1:
            raise ValueError('当前三种解释流程需要带 CLS token 的 ViT 权重；请先训练 configs/vit.yaml。')
        if 'preprocessing' not in self.config:
            raise ValueError('模型缺少预处理配置，请使用当前训练流程生成的 best.pt。')
        self.model.eval()
        for block in self.model.blocks:
            if hasattr(block.attn, 'fused_attn'):
                block.attn.fused_attn = False
        self.transform = build_transforms(self.config['preprocessing'])
        self.source = {'checkpoint_hash': file_hash(checkpoint_path),
                       'manifest_hash': self.provenance['manifest_hash'],
                       'preprocessing': self.config['preprocessing']}
        self.decision = load_decision(decision_path, self.source)
        size = int(self.config['preprocessing']['image_size'])
        patch = min(32, size)
        self.options = {'grid_size': min(7, size), 'lime_samples': 200,
                        'occlusion_patch': patch, 'occlusion_stride': max(1, patch // 2),
                        'weights': (0.5, 0.3, 0.2), **(xai_options or {})}
        self.lock = threading.Lock()

    def info(self):
        return {'name': self.config['model']['name'], 'device': str(self.device),
                'threshold': self.decision['threshold'], 'decision_objective': self.decision['objective'],
                'checkpoint_hash': self.source['checkpoint_hash'], 'image_size': self.config['preprocessing']['image_size'],
                'xai_options': self.options, 'source_status': 'verified',
                'analysis_hash': object_hash({'source': self.source, 'decision': self.decision,
                                              'options': self.options, 'device': str(self.device)})}

    def analyze(self, image_path, output_dir, image_id, progress):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with self.lock:
            with Image.open(image_path) as source:
                image = self.transform.transforms[0](source.convert('RGB'))
            progress('classifying', None)
            probability = float(predict_prob(self.model, self.transform(image).unsqueeze(0), self.device)[0])
            validate_predictions([0], [probability])
            prediction = {'prob_cancer': probability, 'threshold': self.decision['threshold'],
                          'pred_label': int(probability >= self.decision['threshold']),
                          'score_margin': abs(probability - self.decision['threshold'])}
            progress('explaining', prediction)
            seed = 42 + int(object_hash(image_id)[:8], 16) % 1000000
            maps, timings = explain(self.model, image, self.transform, self.device, seed=seed, **self.options)
            np.savez_compressed(output / 'maps.npz', **maps)
            for method in ('attention', 'lime', 'occlusion', 'fusion', 'disagreement'):
                overlay = overlay_heatmap(image, maps[method])
                Image.fromarray(np.uint8(np.clip(overlay * 255, 0, 255))).save(output / f'{method}.png')
            evidence = {'id': image_id, 'model': self.config['model']['name'], **prediction,
                        'target': 'cancer_score', 'source': self.source, 'decision': self.decision,
                        'top_attention_regions': attention_region_summary(maps['attention'])[:3],
                        'top_lime_regions': attention_region_summary(maps['lime'])[:3],
                        'top_occlusion_regions': attention_region_summary(maps['occlusion'])[:3],
                        'top_final_score_regions': attention_region_summary(maps['fusion'])[:3],
                        'fusion_weights': list(self.options['weights']), 'fusion_status': 'exploratory',
                        'mean_method_disagreement': float(maps['disagreement'].mean()),
                        'timings': timings}
            return prediction, evidence
