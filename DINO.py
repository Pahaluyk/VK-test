import json
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path
import random
from PIL import Image


class IntroFrameDataset(Dataset):
    def __init__(self, frame_data, processor, transform=None, augment=False):
        self.frame_data = frame_data
        self.processor = processor
        self.transform = transform
        self.augment = augment
        
    def __len__(self):
        return len(self.frame_data)

    def __getitem__(self, idx):
        frame_path, label = self.frame_data[idx]
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame_pil = Image.fromarray(frame)
        
        inputs = self.processor(images=frame_pil, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0) 
        
        return pixel_values, label


class ImprovedClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)


class IntroDetector:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', dino_model='facebook/dino-vitb16'):
        self.device = device
        self.fps_extract = 2
        self.frame_cache_dir = Path('frame_cache')
        self.frame_cache_dir.mkdir(exist_ok=True)
        self.max_video_seconds = 300

        self.processor = ViTImageProcessor.from_pretrained(dino_model)
        self.feature_extractor = ViTModel.from_pretrained(dino_model).to(self.device)
        self.feature_extractor.eval()

        if 'vits' in dino_model:
            feature_dim = 384
        else:  
            feature_dim = 768
            
        self.classifier = ImprovedClassifier(input_dim=feature_dim).to(self.device)

    def parse_time(self, time_str):
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
        return int(parts[0])

    def extract_frames_from_interval(self, video_path, start_time, end_time, label, video_id):
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        max_end_time = min(end_time, self.max_video_seconds)
        
        start_frame = int(start_time * video_fps)
        end_frame = int(max_end_time * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_interval = int(video_fps / self.fps_extract)
        frame_paths = []
        current_frame = start_frame

        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (current_frame - start_frame) % frame_interval == 0:
                frame_filename = f"{video_id}_{current_frame}_{label}.jpg"
                frame_path = self.frame_cache_dir / frame_filename
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append((str(frame_path), label))
            current_frame += 1

        cap.release()
        return frame_paths

    def create_dataset(self, data_type='train'):
        with open(f'labels_{data_type}.json', 'r', encoding='utf-8') as f:
            labels = json.load(f)

        all_frame_data = []

        for video_id in tqdm(labels, desc="Extracting frames"):
            video_path = f"{data_type}/{video_id}/{video_id}.mp4"
            if not os.path.exists(video_path):
                continue

            start_time = self.parse_time(labels[video_id]['start'])
            end_time = self.parse_time(labels[video_id]['end'])
            if start_time > end_time:
                start_time, end_time = end_time, start_time

            positive_frames = self.extract_frames_from_interval(video_path, start_time, end_time, 1, f"{video_id}_intro")
            all_frame_data.extend(positive_frames)

            cap = cv2.VideoCapture(video_path)
            total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            negative_samples = []
            
            if end_time + 30 < total_duration:
                neg_start = end_time + 30
                neg_end = min(neg_start + (end_time - start_time), total_duration - 10)
                if neg_end > neg_start:
                    negative_samples.append((neg_start, neg_end, f"{video_id}_neg_after"))
            
            if start_time > 60:
                pre_end = start_time - 10
                pre_start = max(10, pre_end - (end_time - start_time))
                negative_samples.append((pre_start, pre_end, f"{video_id}_neg_before"))
            
            if total_duration > 300:
                mid_start = total_duration / 2
                mid_end = min(mid_start + 20, total_duration - 10)
                negative_samples.append((mid_start, mid_end, f"{video_id}_neg_mid"))
            
            for neg_start, neg_end, neg_id in negative_samples:
                neg_frames = self.extract_frames_from_interval(video_path, neg_start, neg_end, 0, neg_id)
                all_frame_data.extend(neg_frames)

        random.shuffle(all_frame_data)
        return all_frame_data

    def get_dino_features(self, pixel_values_batch):
        pixel_values_batch = pixel_values_batch.to(self.device)
        
        with torch.no_grad():
            outputs = self.feature_extractor(pixel_values=pixel_values_batch)
            features = outputs.last_hidden_state[:, 0, :]  
        
        return features

    def train_classifier(self, frame_data, max_epochs=15):
        train_data, val_data = train_test_split(frame_data, test_size=0.2, random_state=42, stratify=[label for _, label in frame_data])

        train_dataset = IntroFrameDataset(train_data, self.processor)
        val_dataset = IntroFrameDataset(val_data, self.processor)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

        labels_array = np.array([label for _, label in train_data])
        pos_weight = len(labels_array) / (2 * (labels_array == 1).sum())
        
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(self.device))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

        best_val_acc = 0
        patience = 5
        no_improve = 0
        
        for epoch in range(max_epochs):
            self.classifier.train()
            train_loss, train_correct, train_total = 0, 0, 0

            for pixel_values, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                labels = labels.float().to(self.device)
                
                features = self.get_dino_features(pixel_values)

                outputs = self.classifier(features).squeeze(-1)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
                train_total += labels.size(0)

            val_correct, val_total = 0, 0
            self.classifier.eval()
            with torch.no_grad():
                for pixel_values, labels in val_loader:
                    labels = labels.float().to(self.device)
                    features = self.get_dino_features(pixel_values)
                    outputs = self.classifier(features).squeeze(-1)
                    val_correct += ((torch.sigmoid(outputs) > 0.5).float() == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = 100 * val_correct / val_total
            scheduler.step(val_acc)
            print(f"Epoch {epoch+1}: Val={val_acc:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': self.classifier.state_dict(),
                    'val_acc': val_acc
                }, 'best_classifier_dino.pth')
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                break

    def predict_video_batch(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frame = int(min(total_frames, fps * self.max_video_seconds))
        frame_interval = int(fps / self.fps_extract)

        predictions, timestamps = [], []
        frame_idx = 0
        batch_frames = []
        batch_indices = []
        BATCH_SIZE = 8  

        self.classifier.eval()
        with torch.no_grad():
            while frame_idx < max_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    
                    inputs = self.processor(images=frame_pil, return_tensors="pt")
                    pixel_values = inputs['pixel_values'].squeeze(0)
                    
                    batch_frames.append(pixel_values)
                    batch_indices.append(frame_idx)
                    
                    if len(batch_frames) == BATCH_SIZE or frame_idx >= max_frame - 1:
                        batch_tensor = torch.stack(batch_frames).to(self.device)
                        features = self.get_dino_features(batch_tensor)
                        outputs = torch.sigmoid(self.classifier(features)).squeeze(-1).cpu().numpy()
                        
                        for i, output in enumerate(outputs):
                            predictions.append(output)
                            timestamps.append(batch_indices[i] / fps)
                        
                        batch_frames = []
                        batch_indices = []

                frame_idx += 1
            
        cap.release()

        if len(predictions) > 5:
            smoothed = []
            window = 5
            for i in range(len(predictions)):
                start = max(0, i - window // 2)
                end = min(len(predictions), i + window // 2 + 1)
                smoothed.append(np.mean(predictions[start:end]))
            predictions = smoothed

        intros = []
        in_intro = False
        intro_start = None
        non_intro_count = 0
        threshold = 0.5

        for i, (prob, ts) in enumerate(zip(predictions, timestamps)):
            is_intro = prob > threshold
            
            if is_intro and not in_intro:
                in_intro = True
                intro_start = ts
                non_intro_count = 0
            elif not is_intro and in_intro:
                non_intro_count += 1
                if non_intro_count >= 8:
                    intro_end = timestamps[i - non_intro_count]
                    if intro_end - intro_start >= 3:
                        intros.append((intro_start, intro_end))
                    in_intro = False
                    non_intro_count = 0
            elif is_intro and in_intro:
                non_intro_count = 0

        if in_intro and len(timestamps) > 0 and timestamps[-1] - intro_start >= 3:
            intros.append((intro_start, timestamps[-1]))

        if len(intros) > 1:
            merged = []
            current_start, current_end = intros[0]
            
            for next_start, next_end in intros[1:]:
                if next_start - current_end < 10:
                    current_end = next_end
                else:
                    merged.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            
            merged.append((current_start, current_end))
            intros = merged

        return intros

    def format_time(self, seconds):
        return f"{int(seconds // 3600)}:{int((seconds % 3600) // 60):02d}:{int(seconds % 60):02d}"

    def process_test_dataset(self):
        with open('labels_test.json', 'r', encoding='utf-8') as f:
            test_labels = json.load(f)

        predictions = {}
        for i, video_id in enumerate(test_labels):
            print(f"[{i+1}/{len(test_labels)}] {video_id}")
            video_path = f"test/{video_id}/{video_id}.mp4"
            if not os.path.exists(video_path):
                predictions[video_id] = {'start': "0:00:05", 'end': "0:00:15"}
                continue
            try:
                intros = self.predict_video_batch(video_path)
                if intros:
                    start, end = max(intros, key=lambda x: x[1] - x[0])
                    predictions[video_id] = {'start': self.format_time(start), 'end': self.format_time(end)}
                else:
                    predictions[video_id] = {'start': "0:00:05", 'end': "0:00:15"}
            except Exception as e:
                predictions[video_id] = {'start': "0:00:05", 'end': "0:00:15"}

        return predictions, test_labels

    def evaluate(self, predictions, ground_truth):
        ious = []
        for video_id in ground_truth:
            if video_id not in predictions:
                ious.append(0.0)
                continue
            true_start = self.parse_time(ground_truth[video_id]['start'])
            true_end = self.parse_time(ground_truth[video_id]['end'])
            if true_start > true_end:
                true_start, true_end = true_end, true_start
            pred_start = self.parse_time(predictions[video_id]['start'])
            pred_end = self.parse_time(predictions[video_id]['end'])
            inter_start = max(true_start, pred_start)
            inter_end = min(true_end, pred_end)
            intersection = max(0, inter_end - inter_start)
            union = (true_end - true_start) + (pred_end - pred_start) - intersection
            ious.append(intersection / union if union > 0 else 0.0)
        return np.mean(ious) if ious else 0.0


def main():
    detector = IntroDetector(dino_model='facebook/dino-vitb16')
    
    if os.path.exists('best_classifier_dino.pth'):
        checkpoint = torch.load('best_classifier_dino.pth', map_location=detector.device)
        if isinstance(checkpoint, dict):
            detector.classifier.load_state_dict(checkpoint['model_state_dict'])
        else:
            detector.classifier.load_state_dict(checkpoint)
    else:
        if os.path.exists('frame_data.pkl'):
            with open('frame_data.pkl', 'rb') as f:
                frame_data = pickle.load(f)
        else:
            frame_data = detector.create_dataset('train')
            with open('frame_data.pkl', 'wb') as f:
                pickle.dump(frame_data, f)
        detector.train_classifier(frame_data, max_epochs=15)

    test_predictions, test_labels = detector.process_test_dataset()
    test_iou = detector.evaluate(test_predictions, test_labels)
    print(f"\nIoU: {test_iou:.3f}")
    
    with open('results_dino.json', 'w', encoding='utf-8') as f:
        json.dump(test_predictions, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()