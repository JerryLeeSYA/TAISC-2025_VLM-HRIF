#!/usr/bin/env python3
"""
流程: 訓練 -> 預測 -> GMM轉換 -> 正規化 -> 產出提交檔案
模型: ViT-Large-336, 圖像尺寸: 336x336
"""

import os, platform
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, List
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from huggingface_hub import snapshot_download

# 安全導入模組
try:
    from transformers import CLIPModel, CLIPProcessor
    CLIP_AVAILABLE = True
    print("✅ CLIP模型可用")
except ImportError:
    CLIP_AVAILABLE = False
    print("❌ CLIP模型不可用")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture # 新增 GMM 導入
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm

warnings.filterwarnings("ignore")

class Config:
    """配置類"""
    
    def _find_available_model(self):
        target_root = Path("./models")
        target_root.mkdir(exist_ok=True)
        local_model_path_str = "./models/openai--clip-vit-large-patch14-336"
        self.local_model_paths = {"clip-large": local_model_path_str}
        
        search_dirs = [local_model_path_str, str(target_root)]
        for base in search_dirs:
            for root, _, files in os.walk(base):
                if "config.json" in files:
                    print(f"✅ 找到本地模型: {root}")
                    return root

        repo_id = "openai/clip-vit-large-patch14-336"
        is_windows = platform.system().lower() == "windows"
        if is_windows:
            os.environ["HF_HUB_DISABLE_SYMLINKS_WINDOWS"] = "1"

        try:
            print(f"⚠️ 本地未找到，開始下載 {repo_id} → {target_root} …")
            local_path = snapshot_download(
                repo_id=repo_id,
                local_dir=target_root / repo_id.replace("/", "--"),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print(f"✅ 下載完成，已存放於: {local_path}")
            return local_path
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
        return None
    
    def __init__(self):
        # --- 模型 ---
        self.model_name = self._find_available_model()
        self.model_type = "clip"
        
        # --- 核心訓練參數 ---
        self.danger_class_weight = 1.5
        self.use_class_weights = True
        self.prediction_threshold = 0.35
        self.image_size = 336
        self.max_length = 77
        self.batch_size = 4
        self.learning_rate = 1e-5
        self.weight_decay = 0.001
        self.max_grad_norm = 0.05
        self.num_epochs = 1
        self.early_stopping_patience = 8
        self.monitor_metric = "val_f1"
        self.min_delta = 0.01
        
        # --- 路徑與環境 ---
        self.data_path = "./AVA Dataset"
        self.output_dir = "./outputs"
        self.checkpoint_dir = "./checkpoints"
        self.random_seed = 42
        self.train_split = 0.8
        self.val_split = 0.2
        self.save_best_only = True
        
        # --- 打印配置 ---
        print("\n" + "="*50)
        print("🎯 配置已鎖定")
        print(f"  - 模型名稱: {self.model_name}")
        print(f"  - 數據路徑: {self.data_path}")
        print(f"  - 學習率: {self.learning_rate}")
        print(f"  - 批次大小: {self.batch_size}")
        print(f"  - 訓練輪數: {self.num_epochs}")
        print("="*50 + "\n")


class TAISCDataset(Dataset):
    """TAISC數據集類"""
    def __init__(self, data_path: str, split: str = "train", processor=None, config=None):
        self.data_path = Path(data_path)
        self.split = split
        self.processor = processor
        self.config = config
        
        self.danger_prompts = ["A dash-cam frame where a crash is about to occur","Vehicle cutting in sharply, high risk of collision","Car hard-braking, following distance almost zero","Pedestrian suddenly entering the driving lane","Bike crossing ahead, imminent impact expected","Oncoming traffic encroaching the ego lane","Obstacle on road, driver must swerve within seconds","Red-light runner entering intersection, crash likely","Ego vehicle tailgating, no safe braking margin","Rain-soaked road and skid about to happen","Vehicle drifting across lane markings toward us","Blind curve with blocked sight, head-on risk","Truck merging without gap, emergency response needed","High speed plus stalled car ahead, unavoidable danger","Crowd jaywalking, collision risk very high","Stop-and-go traffic, abrupt rear-end threat","Sharp turn taken too fast, loss of control imminent","Road debris directly in driving path, impact imminent","Driver texting, vehicle veering off lane","Critical split-second before traffic accident"]
        self.safety_prompts = ["Calm highway driving with ample following distance","Urban street scene, vehicles obeying traffic rules","Smooth traffic flow, no immediate hazards detected","Ego vehicle centered in lane, speed appropriate","Clear weather and empty road ahead","Stop line respected, pedestrians safely crossing","Cruise control on straight freeway, stable situation","All vehicles maintaining lane discipline and gaps","Green-light passage with no conflicting traffic","Daytime driving, excellent visibility, no risks","Motorists signalling properly before lane change","Roundabout navigation executed safely","Low speed residential area, no obstacles present","Car ahead braking gently, safe distance kept","Driver scanning mirrors, environment under control","No pedestrians or cyclists near path of travel","Dry pavement, tyres gripping well, stable handling","Night driving with correct headlight use, clear road","Ego car waiting patiently at crosswalk, safe","Routine commute traffic, normal conditions"]
        self.neutral_prompts = ["Describe the traffic situation in this frame","What potential hazards can be seen here?","Analyse the driver’s required reaction","Explain the safety level of this road scene","Summarise the dynamic elements in view","Classify whether emergency action is needed"]
        
        self.samples = self._load_samples()
        print(f"載入 {split} 數據: {len(self.samples)} 個樣本")

    def _load_samples(self) -> List[Dict]:
        samples = []
        data_root = self.data_path
        
        if self.split in ["train", "val"]:
            for scene in ["road", "freeway"]:
                csv_path = data_root / f"{scene}_train.csv"
                if not csv_path.exists(): continue
                
                df = pd.read_csv(csv_path)
                scene_dir = data_root / scene / "train"
                
                for _, row in df.iterrows():
                    sequence_path = scene_dir / row['file_name']
                    if not sequence_path.exists(): continue
                    
                    image_files = sorted(list(sequence_path.glob('*.jpg')) + list(sequence_path.glob('*.png')))
                    if image_files:
                        samples.append({
                            'image_path': str(image_files[0]), 'file_name': row['file_name'],
                            'scene': scene, 'risk': int(row['risk'])
                        })
        elif self.split == "test":
            for scene in ["road", "freeway"]:
                scene_dir = data_root / scene / "test"
                if not scene_dir.exists(): continue
                
                for sequence_path in scene_dir.iterdir():
                    if not sequence_path.is_dir(): continue
                    image_files = sorted(list(sequence_path.glob('*.jpg')) + list(sequence_path.glob('*.png')))
                    if image_files:
                        samples.append({
                            'image_path': str(image_files[0]), 'file_name': sequence_path.name,
                            'scene': scene, 'risk': 0
                        })
        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['image_path']).convert('RGB').resize(
                (self.config.image_size, self.config.image_size), Image.Resampling.LANCZOS)
            
            risk_label = sample['risk']
            rand_val = random.random()

            if risk_label == 1:
                if rand_val < 0.8: prompt = random.choice(self.danger_prompts)
                elif rand_val < 0.9: prompt = random.choice(self.safety_prompts)
                else: prompt = random.choice(self.neutral_prompts)
            else:
                if rand_val < 0.8: prompt = random.choice(self.safety_prompts)
                elif rand_val < 0.9: prompt = random.choice(self.danger_prompts)
                else: prompt = random.choice(self.neutral_prompts)
            
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt", padding="max_length",
                truncation=True, max_length=self.config.max_length
            )
            for key in inputs.keys():
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].squeeze(0)
            
            return {
                'inputs': inputs, 'label': torch.tensor(sample['risk'], dtype=torch.long),
                'file_name': sample['file_name'], 'scene': sample['scene']
            }
        except Exception as e:
            print(f"處理樣本 {sample['file_name']} 時出錯: {e}")
            return self._get_dummy_sample()

    def _create_dummy_inputs(self):
        return {
            'pixel_values': torch.randn(3, self.config.image_size, self.config.image_size),
            'input_ids': torch.randint(0, 1000, (self.config.max_length,)),
            'attention_mask': torch.ones(self.config.max_length)
        }
        
    def _get_dummy_sample(self):
        return {
            'inputs': self._create_dummy_inputs(), 'label': torch.tensor(0, dtype=torch.long),
            'file_name': 'dummy', 'scene': 'road'
        }

class AccidentPredictionVLM(nn.Module):
    """事故預測VLM模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vlm_model = CLIPModel.from_pretrained(config.model_name, local_files_only=True)
        self.feature_dim = self.vlm_model.config.projection_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, inputs):
        features = self.vlm_model(**inputs).image_embeds
        if self.training:
            features = F.dropout(features, p=0.3, training=True)
        return self.classifier(features).squeeze(1)

class Trainer:
    """訓練器類"""
    def __init__(self, config: Config):
        self.config = config
        self.setup_environment()
        self.setup_model_and_data()
        self.setup_training()
        
    def setup_environment(self):
        self.set_seed(self.config.random_seed)
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"✅ 使用GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("✅ 使用CPU")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def setup_model_and_data(self):
        self.processor = CLIPProcessor.from_pretrained(self.config.model_name, local_files_only=True)
        
        full_dataset = TAISCDataset(self.config.data_path, "train", self.processor, self.config)
        if len(full_dataset) == 0:
            raise ValueError(f"數據集為空！請檢查路徑: {self.config.data_path}")
        
        train_samples, val_samples = train_test_split(
            full_dataset.samples, test_size=self.config.val_split,
            random_state=self.config.random_seed, stratify=[s['risk'] for s in full_dataset.samples]
        )
        
        train_dataset = TAISCDataset(self.config.data_path, "train", self.processor, self.config)
        train_dataset.samples = train_samples
        
        val_dataset = TAISCDataset(self.config.data_path, "val", self.processor, self.config)
        val_dataset.samples = val_samples
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_fn)
        
        self.model = AccidentPredictionVLM(self.config).to(self.device)

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        
        inputs_list = [item['inputs'] for item in batch]
        labels = torch.stack([item['label'] for item in batch])
        file_names = [item['file_name'] for item in batch]
        
        batch_inputs = {}
        for key in inputs_list[0].keys():
            batch_inputs[key] = torch.stack([inp[key] for inp in inputs_list])
            
        return {'inputs': batch_inputs, 'labels': labels, 'file_names': file_names}
    
    def setup_training(self):
        labels = [s['risk'] for s in self.train_loader.dataset.samples]
        pos_w_value = (len(labels) - sum(labels)) / sum(labels) if sum(labels) > 0 else 1.0
        pos_w = torch.tensor([pos_w_value], device=self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        
        for param in self.model.vlm_model.parameters():
            param.requires_grad = False
        
        unfreeze_layers = ["visual.transformer.resblocks.10", "visual.transformer.resblocks.11", "visual.ln_post"]
        for name, param in self.model.vlm_model.named_parameters():
            if any(layer_name in name for layer_name in unfreeze_layers):
                param.requires_grad = True
        
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        
        classifier_params = {'params': self.model.classifier.parameters(), 'lr': 0.000777}
        vision_params_gen = (p for n, p in self.model.vlm_model.named_parameters() if p.requires_grad)
        vision_finetune_params = {'params': vision_params_gen, 'lr': 1e-5}
        
        self.optimizer = AdamW([classifier_params, vision_finetune_params], weight_decay=self.config.weight_decay)
        self.scheduler = None # 不使用調度器
        
        self.best_metric = 0.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss, all_labels, all_predictions = 0.0, [], []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}", leave=False)
        
        for batch in progress_bar:
            if batch is None: continue
            inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.float())
            
            if torch.isnan(loss): continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > self.config.prediction_threshold).long()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().detach().numpy())
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        if not all_labels: return 0.0, 0.0, 0.0
        avg_loss = total_loss / len(self.train_loader)
        train_acc = accuracy_score(all_labels, all_predictions)
        train_f1 = f1_score(all_labels, all_predictions, zero_division=0)
        return avg_loss, train_acc, train_f1
    
    def validate(self):
        self.model.eval()
        total_loss, all_labels, all_probabilities = 0, [], []
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                if batch is None: continue
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                labels = batch['labels'].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                total_loss += loss.item()
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(torch.sigmoid(outputs).cpu().numpy())
        
        if not all_probabilities: return float('inf'), 0.0, 0.0, 0.5
        
        val_loss = total_loss / len(self.val_loader)
        val_auc = roc_auc_score(all_labels, all_probabilities) if len(set(all_labels)) > 1 else 0.5
        
        best_f1, best_acc, best_threshold = 0, 0, 0.5
        for threshold in np.arange(0.3, 0.7, 0.05):
            predictions = (np.array(all_probabilities) > threshold).astype(int)
            f1 = f1_score(all_labels, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1, best_acc, best_threshold = f1, accuracy_score(all_labels, predictions), threshold
                
        self.config.prediction_threshold = best_threshold
        return val_loss, best_acc, best_f1, val_auc
        
    def save_checkpoint(self, epoch, is_best=False):
        state = {'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict(), 'best_metric': self.best_metric}
        if is_best:
            torch.save(state, os.path.join(self.config.checkpoint_dir, 'best_model.pth'))
            print(f"💾 新最佳模型已保存 (監控指標: {self.best_metric:.4f})")
    
    def train(self):
        for epoch in range(self.config.num_epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(epoch)
            val_loss, val_acc, val_f1, val_auc = self.validate()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs} | "
                  f"Train: Loss={train_loss:.4f} Acc={train_acc:.4f} F1={train_f1:.4f} | "
                  f"Val: Loss={val_loss:.4f} Acc={val_acc:.4f} F1={val_f1:.4f} AUC={val_auc:.4f}")
            
            current_metric = locals()[self.config.monitor_metric]
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, is_best)
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"🛑 早停！已連續 {self.config.early_stopping_patience} 個 epoch 無改善。")
                break
        print(f"\n🎉 訓練完成！最佳 {self.config.monitor_metric}: {self.best_metric:.4f}")

class Predictor:
    """預測器類"""
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AccidentPredictionVLM(self.config)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict']) # <--- 從字典中取出權重
        self.model.to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(config.model_name, local_files_only=True)
        print(f"✅ 預測器載入模型: {checkpoint_path}")

    def collate_fn(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        inputs_list = [item['inputs'] for item in batch]
        file_names = [item['file_name'] for item in batch]
        batch_inputs = {key: torch.stack([inp[key] for inp in inputs_list]) for key in inputs_list[0]}
        return {'inputs': batch_inputs, 'file_names': file_names}

    def predict_test_data(self) -> pd.DataFrame:
        test_dataset = TAISCDataset(self.config.data_path, "test", self.processor, self.config)
        if len(test_dataset) == 0: return pd.DataFrame()
        
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0, collate_fn=self.collate_fn)
        predictions = {}
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                if batch is None: continue
                inputs = {k: v.to(self.device) for k, v in batch['inputs'].items()}
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs).cpu().numpy()
                for file_name, prob in zip(batch['file_names'], probabilities):
                    predictions[file_name] = float(prob)
        
        submission_df = pd.DataFrame(list(predictions.items()), columns=['file_name', 'risk'])
        raw_output_path = os.path.join(self.config.output_dir, 'submission_raw.csv')
        submission_df.to_csv(raw_output_path, index=False)
        print(f"✅ 原始預測已保存至: {raw_output_path}")
        return submission_df

# ===================================================================
# 後處理函數
# ===================================================================
def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """將分數進行Min-Max正規化到[0, 1]區間"""
    print("📏 步驟 3a: 執行Min-Max正規化...")
    min_val, max_val = scores.min(), scores.max()
    if max_val == min_val:
        return np.full_like(scores, 0.5) # 如果所有值都一樣，返回0.5
    normalized = (scores - min_val) / (max_val - min_val)
    return normalized

def find_gmm_threshold(scores: np.ndarray, config: Config) -> float | None:
    """
    使用 2-component Gaussian Mixture 找決策邊界（posterior=0.5）。
    """
    scores_reshaped = scores.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gmm = GaussianMixture(n_components=2, random_state=config.random_seed)
        gmm.fit(scores_reshaped)
    
    means = gmm.means_.flatten()
    # 如果兩個分佈的中心點太接近，視為單峰，GMM不適用
    if abs(means[0] - means[1]) < 1e-3:
        return None

    # 解 w1*N1 = w2*N2 的二次方程式，找到兩高斯分佈的交點
    m1, m2 = means
    v1, v2 = gmm.covariances_.flatten()
    w1, w2 = gmm.weights_
    
    a = 1/(2*v1) - 1/(2*v2)
    b = m2/(v2) - m1/(v1)
    c = m1**2/(2*v1) - m2**2/(2*v2) + np.log((w2*np.sqrt(v1))/(w1*np.sqrt(v2)))
    
    # 忽略虛根，選擇落在分數區間內的實根
    roots = np.roots([a, b, c])
    real_roots = [r.real for r in roots if r.imag == 0]
    valid_thresholds = [t for t in real_roots if scores.min() < t < scores.max()]
    
    return valid_thresholds[0] if valid_thresholds else None

def find_best_threshold(scores: np.ndarray, config: Config, fallback_percentile=80) -> tuple[float, str]:
    """
    優先嘗試GMM尋找閾值，若失敗則退回使用百分位法。
    返回 (閾值, 使用的方法名稱)。
    """
    # 嘗試 GMM
    gmm_thr = find_gmm_threshold(scores, config)
    if gmm_thr is not None:
        return gmm_thr, "GMM"
    
    # GMM 失敗，使用百分位法
    percentile_thr = np.percentile(scores, fallback_percentile)
    return percentile_thr, f"Percentile_{fallback_percentile}"

# ===================================================================
# 修改後的主函數
# ===================================================================

def main():
    """主函數：訓練 -> 預測 -> 正規化 -> GMM閾值 -> 二元化 -> 產出"""
    try:
        # --- 步驟 0: 初始化配置 ---
        config = Config()

        # --- 步驟 1: 訓練模型 ---
        print("\n--- 步驟 1: 開始訓練 ---")
        trainer = Trainer(config)
        trainer.train()

        # --- 步驟 2: 使用最佳模型進行預測 ---
        print("\n--- 步驟 2: 使用最佳模型進行預測 ---")
        best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"訓練結束但找不到最佳模型文件: {best_model_path}")
            
        predictor = Predictor(config, best_model_path)
        submission_df = predictor.predict_test_data()

        if submission_df.empty:
            print("❌ 預測結果為空，無法進行後處理。")
            return

        # --- 步驟 3: 後處理 (正規化 -> GMM閾值 -> 二元化) ---
        print("\n--- 步驟 3: 進行後處理 ---")
        
        # 步驟 3a: 對所有風險分數進行正規化
        raw_scores = submission_df['risk'].values
        submission_df['normalized_risk'] = normalize_scores(raw_scores)
        
        # 步驟 3b: 分場景計算GMM閾值
        print("🔮 步驟 3b: 分場景計算GMM閾值...")
        submission_df['group'] = submission_df['file_name'].apply(
            lambda x: 'road' if x.startswith('road') else 'freeway')
            
        thresholds = {}
        for group in ['road', 'freeway']:
            group_scores = submission_df[submission_df['group'] == group]['normalized_risk'].values
            if len(group_scores) > 0:
                thr, method = find_best_threshold(group_scores, config)
                thresholds[group] = thr
                print(f"  - {group.upper():7s} -> 方法: {method:12s} | 閾值: {thr:.4f}")
            else:
                thresholds[group] = 0.5 # 如果沒有該場景的數據，預設一個閾值
        
        # 步驟 3c: 應用閾值進行二元化
        print("🚦 步驟 3c: 應用閾值生成 0/1 標籤...")
        submission_df['risk'] = submission_df.apply(
            lambda row: 1 if row['normalized_risk'] >= thresholds[row['group']] else 0,
            axis=1
        )

        # --- 步驟 4: 產生最終提交檔案 ---
        print("\n--- 步驟 4: 產生最終提交檔案 ---")
        final_submission_df = submission_df[['file_name', 'risk']]
        
        final_output_path = os.path.join(config.output_dir, 'submission_final.csv')
        final_submission_df.to_csv(final_output_path, index=False)
        
        print("\n" + "="*60)
        print("🎉🎉🎉 流程執行完畢！🎉🎉🎉")
        print(f"最終提交檔案已保存至: {final_output_path}")
        print("最終預測分佈:")
        print(final_submission_df['risk'].value_counts().rename(index={0: '低風險 (0)', 1: '高風險 (1)'}))
        print("="*60)

    except KeyboardInterrupt:
        print("\n⏹️ 程序被用戶中斷。")
    except Exception as e:
        print(f"\n❌ 執行過程中出現嚴重錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()