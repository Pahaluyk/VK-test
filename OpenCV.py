import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import time
import cv2
class ImprovedIntroDetector:  
    def __init__(self):
        self.frame_interval = 0.5  
        self.max_intro_length = 300
        self.min_intro_length = 8   
        
    def extract_enhanced_features(self, video_path: str) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            cap.release()
            return None
            
        frame_step = max(1, int(fps * self.frame_interval))
        max_frames = int(self.max_intro_length / self.frame_interval)
        
        features = []
        frame_count = 0
        processed_frames = 0
        
        while processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_step == 0:
                medium_frame = cv2.resize(frame, (64, 64))
                gray = cv2.cvtColor(medium_frame, cv2.COLOR_BGR2GRAY)
                
                mean_b = np.mean(medium_frame[:, :, 0])
                mean_g = np.mean(medium_frame[:, :, 1]) 
                mean_r = np.mean(medium_frame[:, :, 2])
                
                std_b = np.std(medium_frame[:, :, 0])
                std_g = np.std(medium_frame[:, :, 1])
                std_r = np.std(medium_frame[:, :, 2])
                
                brightness = np.mean(gray)
                contrast = np.std(gray)
                
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges) / (64 * 64 * 255)
                
                hist = cv2.calcHist([gray], [0], None, [8], [0, 256])
                hist_norm = hist.flatten() / (hist.sum() + 1e-8)
                
                feature_vector = [mean_r, mean_g, mean_b, std_r, std_g, std_b, 
                                brightness, contrast, edge_density] + hist_norm.tolist()
                
                features.append(feature_vector)
                processed_frames += 1
                
            frame_count += 1
            
        cap.release()
        
        if len(features) < self.min_intro_length / self.frame_interval:
            return None
            
        return np.array(features)
    
    def find_intro_improved(self, features: np.ndarray) -> Optional[Tuple[float, float]]:
        if len(features) < self.min_intro_length / self.frame_interval:
            return None
        
        min_length_frames = int(self.min_intro_length / self.frame_interval)
        max_length_frames = min(len(features), int(30 / self.frame_interval))
        max_start_frames = min(len(features) - min_length_frames, int(20 / self.frame_interval))
        
        best_candidates = []
        
        for start in range(max_start_frames):
            for length in range(min_length_frames, min(max_length_frames, len(features) - start + 1)):
                sequence = features[start:start + length]
                
                stabilities = []
                
                for feature_idx in range(sequence.shape[1]):
                    feature_values = sequence[:, feature_idx]
                    if len(feature_values) > 1 and np.mean(feature_values) > 1e-6:
                        cv = np.std(feature_values) / (np.mean(feature_values) + 1e-6)
                        stability = 1 / (cv + 0.01)
                        stabilities.append(stability)
                
                if not stabilities:
                    continue
                
                avg_stability = np.mean(stabilities)
                position_bonus = max(0, 1 - start / 10)
                
                length_seconds = length * self.frame_interval
                ideal_length = 12  
                length_bonus = 1 - abs(length_seconds - ideal_length) / ideal_length
                length_bonus = max(0, length_bonus)
                
                if length_seconds < 6:
                    length_penalty = 0.5
                elif length_seconds > 25:
                    length_penalty = 0.7
                else:
                    length_penalty = 1.0
                
                total_score = avg_stability * (1 + position_bonus * 0.3 + length_bonus * 0.4) * length_penalty
                
                best_candidates.append({
                    'start': start,
                    'length': length,
                    'score': total_score,
                    'stability': avg_stability,
                    'position_bonus': position_bonus,
                    'length_bonus': length_bonus,
                    'length_seconds': length_seconds
                })
        
        if not best_candidates:
            return None
        
        best_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        best = best_candidates[0]
        threshold = 3.0  
        
        if best['score'] > threshold:
            start_time = best['start'] * self.frame_interval
            end_time = (best['start'] + best['length']) * self.frame_interval
            return (start_time, end_time)
        
        reasonable_candidates = [c for c in best_candidates if c['score'] > 1.5 and 8 <= c['length_seconds'] <= 20]
        
        if reasonable_candidates:
            fallback = reasonable_candidates[0]
            start_time = fallback['start'] * self.frame_interval
            end_time = (fallback['start'] + fallback['length']) * self.frame_interval
            return (start_time, end_time)
        
        return None
    
    def detect_intro(self, video_path: str) -> Optional[Tuple[float, float]]:
        try:
            features = self.extract_enhanced_features(video_path)
            if features is None:
                return None
                
            result = self.find_intro_improved(features)
            
            if result is None:
                return (5.0, 15.0)
            
            return result
            
        except Exception as e:
            return None

def parse_time_to_seconds(time_str: str) -> float:
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    else:
        return float(parts[0])

def calculate_iou(pred_start: float, pred_end: float, 
                  true_start: float, true_end: float) -> float:
    intersection_start = max(pred_start, true_start)
    intersection_end = min(pred_end, true_end)
    
    if intersection_start >= intersection_end:
        return 0.0
        
    intersection = intersection_end - intersection_start
    union = (pred_end - pred_start) + (true_end - true_start) - intersection
    
    return intersection / union if union > 0 else 0.0

def process_dataset() -> Dict:
    data_path = "D:/ВК"
    split = "test"
    max_videos = 45
    
    data_path = Path(data_path)
    detector = ImprovedIntroDetector()
    
    split_dir = data_path / split
    labels_in_split = split_dir / 'labels.json'
    labels_with_split_name = split_dir / f'labels_{split}.json'
    labels_in_root = data_path / f'labels_{split}.json'
    
    labels_file = None
    if labels_in_split.exists():
        labels_file = labels_in_split
    elif labels_with_split_name.exists():
        labels_file = labels_with_split_name
    elif labels_in_root.exists():
        labels_file = labels_in_root
    else:
        raise FileNotFoundError(f"Файл разметки не найден")
        
    with open(labels_file, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    if max_videos:
        ground_truth = dict(list(ground_truth.items())[:max_videos])
    
    predictions = {}
    processing_times = {}
    
    for i, (video_id, gt_data) in enumerate(ground_truth.items()):
        print(f"[{i+1}/{len(ground_truth)}] {video_id}")
        
        video_dir = split_dir / video_id
        video_path = None
        
        if video_dir.exists():
            for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                candidate_path = video_dir / f"{video_id}{ext}"
                if candidate_path.exists():
                    video_path = str(candidate_path)
                    break
            
            if not video_path:
                for file_path in video_dir.iterdir():
                    if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                        video_path = str(file_path)
                        break
        
        if video_path and os.path.exists(video_path):
            start_time = time.time()
            result = detector.detect_intro(video_path)
            processing_time = time.time() - start_time
            
            predictions[video_id] = result
            processing_times[video_id] = processing_time
        else:
            predictions[video_id] = None
            processing_times[video_id] = 0
    
    ious = []
    detected_count = 0
    temporal_errors = []
    
    for video_id, gt_data in ground_truth.items():
        if video_id in predictions and predictions[video_id] is not None:
            pred_start, pred_end = predictions[video_id]
            true_start = parse_time_to_seconds(gt_data['start'])
            true_end = parse_time_to_seconds(gt_data['end'])
            
            iou = calculate_iou(pred_start, pred_end, true_start, true_end)
            ious.append(iou)
            
            if iou > 0.3:
                detected_count += 1
                
            start_error = abs(pred_start - true_start)
            end_error = abs(pred_end - true_end)
            temporal_errors.append((start_error + end_error) / 2)
        else:
            ious.append(0.0)
    
    total_videos = len(ground_truth)
    processed_videos = len([p for p in predictions.values() if p is not None])
    
    metrics = {
        'mean_iou': float(np.mean(ious)) if ious else 0.0,
        'detection_rate': detected_count / total_videos if total_videos > 0 else 0.0,
        'mean_temporal_error': float(np.mean(temporal_errors)) if temporal_errors else 0.0,
        'total_videos': total_videos,
        'processed_videos': processed_videos,
        'detected_videos': detected_count,
        'total_processing_time': sum(processing_times.values()),
        'avg_processing_time': np.mean(list(processing_times.values())) if processing_times else 0.0
    }
    
    serializable_predictions = {}
    for video_id, pred in predictions.items():
        if pred is not None:
            serializable_predictions[video_id] = {
                'start_time': float(pred[0]),
                'end_time': float(pred[1]),
                'start_formatted': f"0:{int(pred[0]//60):02d}:{int(pred[0]%60):02d}",
                'end_formatted': f"0:{int(pred[1]//60):02d}:{int(pred[1]%60):02d}"
            }
        else:
            serializable_predictions[video_id] = None
    
    return {
        'predictions': serializable_predictions,
        'metrics': metrics,
        'ground_truth': ground_truth,
        'processing_times': processing_times
    }

def main():
    try:
        results = process_dataset()
        
        metrics = results['metrics']
        print(f"\nРезультаты:")
        print(f"IoU: {metrics['mean_iou']:.3f}")
        print(f"Найдено: {metrics['detected_videos']}/{metrics['total_videos']}")
        print(f"Время: {metrics['total_processing_time']:.1f}с")
        
        with open('results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Сохранено в results.json")
            
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())