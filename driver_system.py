#!/usr/bin/env python
# coding: utf-8

# # Akıllı Sürücü İzleme Sistemi 
# 
# Bu proje; kamera görüntüsü üzerinden sürücünün **uykulu olma**, **dikkat dağınıklığı**, **telefon kullanımı**, **göz yönü**, ve **sürücü kimliği uyuşmazlığı** gibi durumlarını tespit etmeyi amaçlar.
# 
# Sistem:
# - MediaPipe FaceMesh ile göz açıklığı, iris yönü ve kafa pozu (yaw/pitch/roll) çıkarır
# - Haarcascade ile yüz ROI bulur
# - Basit kural tabanlı yorgunluk/uyku alarmı üretir
# - YOLOv5 + MediaPipe Hands ile telefon kullanımını yakalamaya çalışır
# - Kritik durumda Web API’ye acil durum tetikler; Web çalışmazsa Telegram’a direkt fallback mesajı yollar
# - Oturum verilerini JSON olarak `sessions/` klasörüne kaydeder
# 
# Notebook yapısı: Kodlar modüllere ayrılmıştır ve en altta `baslat_sistemi_konsoldan()` ile sistem başlatılır.
# 

# In[14]:



# Importlar + Ortam değişkenleri (ENV)

import os

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

import cv2
import numpy as np
import pickle
from collections import deque
import time
import json
from datetime import datetime
import threading

EMERGENCY_URL = os.getenv("EMERGENCY_URL", "http://127.0.0.1:5000/api/emergency/trigger").strip()
EMERGENCY_EYES_CLOSED_SECONDS = float(os.getenv("EMERGENCY_EYES_CLOSED_SECONDS", "10"))
EMERGENCY_COOLDOWN_SECONDS = float(os.getenv("EMERGENCY_COOLDOWN_SECONDS", "120"))

# --- Telegram 
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
TELEGRAM_DIRECT_FALLBACK_ENABLED = os.getenv("TELEGRAM_DIRECT_FALLBACK_ENABLED", "0").strip() == "1"

# Zorunlu alan kontrolü
if TELEGRAM_DIRECT_FALLBACK_ENABLED:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("TELEGRAM_DIRECT_FALLBACK_ENABLED=1 ama TELEGRAM_BOT_TOKEN yok (setx yaptın mı, yeni terminal açtın mı?)")
    if not TELEGRAM_CHAT_ID:
        raise RuntimeError("TELEGRAM_DIRECT_FALLBACK_ENABLED=1 ama TELEGRAM_CHAT_ID yok")

# requests

try:
    import requests  # type: ignore
except Exception as e:
    requests = None
    print(f"[UYARI] requests yuklenemedi: {e}. WEB tetikleyici devre disi.")
try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("[UYARI] mediapipe yüklü değil (mp=None). FaceMesh/Hands çalışmaz.")
try:
    import pygame
    pygame.mixer.init()
    print("[OK] Ses sistemi (pygame) baslatildi")
except Exception as e:
    pygame = None
    print("[UYARI] Ses sistemi baslatilamadi:", e)



# In[15]:



# Telegram'a direkt mesaj (fallback)

def send_telegram_direct(message: str) -> bool:
    """
    Web çalışmıyorsa veya hata veriyorsa, doğrudan Telegram'a mesaj atar.
    TELEGRAM_DIRECT_FALLBACK_ENABLED=1 ise aktif olur.
    """
    enabled = os.environ.get("TELEGRAM_DIRECT_FALLBACK_ENABLED", "0").strip() == "1"
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not enabled:
        return False
    if not token or not chat_id:
        print("[TELEGRAM] Token veya chat_id eksik.")
        return False
    if requests is None:
        print("[TELEGRAM] requests yok -> mesaj atılamadı.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}

    try:
        r = requests.post(url, json=payload, timeout=10)
        print("[TELEGRAM]", r.status_code, r.text[:200])
        return (200 <= r.status_code < 300)
    except Exception as e:
        print("[TELEGRAM] HATA:", e)
        return False



# Web tetikleyici + Telegram fallback  
def trigger_emergency_to_web_async(reason: str, seconds: float, session_filename: str = None, extra: dict = None):
    """
    Driver -> Web'e acil durum sinyali gönderir (async).
    Web tarafı Telegram mesajı gönderebilir.
    Eğer WEB çalışmazsa/yanıt vermezse Telegram DIRECT fallback devreye girer.
    """
    payload = {
        "reason": str(reason),
        "seconds": float(seconds),
        "session_filename": session_filename,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    if extra and isinstance(extra, dict):
        payload["extra"] = extra

    def _notify_fallback(tag: str, err: str = ""):
        msg = (
            f"🚨 ACIL DURUM ({tag})\n"
            f"Neden: {payload['reason']}\n"
            f"Süre: {payload['seconds']:.1f}s\n"
            f"Zaman: {payload['ts']}"
        )
        if payload.get("session_filename"):
            msg += f"\nSession: {payload['session_filename']}"
        if err:
            msg += f"\nHata: {err}"
        send_telegram_direct(msg)

    def _worker():
        if requests is None:
            print("[EMERGENCY->WEB] requests yok -> web'e gonderilemedi. Telegram fallback denenecek.")
            _notify_fallback("WEB_YOK", "requests import edilemedi")
            return

        try:
            r = requests.post(EMERGENCY_URL, json=payload, timeout=10)
            print("[EMERGENCY->WEB]", r.status_code, r.text[:200])

            # Web başarısızsa Telegram fallback
            if r.status_code < 200 or r.status_code >= 300:
                _notify_fallback(f"WEB_{r.status_code}", r.text[:200])

        except Exception as e:
            print("[EMERGENCY->WEB] HATA:", e)
            _notify_fallback("WEB_BAGLANTI_HATA", str(e))

    threading.Thread(target=_worker, daemon=True).start()
    return True


# In[16]:


# Sürücü tanıma: preprocess + çoklu kare enroll

class DriverIdentifier:
    def __init__(self, profile_path="driver_profile.npy", sim_threshold=0.82):
        self.profile_path = profile_path
        self.sim_threshold = sim_threshold
        self.ref_vec = None
        self._load_profile()

    def _load_profile(self):
        if os.path.exists(self.profile_path):
            try:
                data = np.load(self.profile_path, allow_pickle=True)
                self.ref_vec = data.item().get("ref_vec") if isinstance(data.item(), dict) else data
                print(f"[OK] Ana sürücü profili yüklendi: {self.profile_path}")
            except Exception as e:
                print(f"[UYARI] Profil okunamadı: {e}")
                self.ref_vec = None

    def _save_profile(self):
        if self.ref_vec is not None:
            np.save(self.profile_path, {"ref_vec": self.ref_vec})
            print(f"[OK] Ana sürücü profili kaydedildi: {self.profile_path}")

    def _preprocess_face(self, face_gray: np.ndarray) -> np.ndarray:
        fg = face_gray.copy()
        h, w = fg.shape[:2]
        cx1 = int(w * 0.15)
        cx2 = int(w * 0.85)
        cy1 = int(h * 0.15)
        cy2 = int(h * 0.90)
        if cx2 > cx1 and cy2 > cy1:
            fg = fg[cy1:cy2, cx1:cx2]

        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            fg = clahe.apply(fg)
        except Exception:
            try:
                fg = cv2.equalizeHist(fg)
            except Exception:
                pass

        try:
            fg = cv2.GaussianBlur(fg, (3, 3), 0)
        except Exception:
            pass

        return fg

    def _face_to_vec(self, face_gray: np.ndarray):
        try:
            face_gray = self._preprocess_face(face_gray)
            resized = cv2.resize(face_gray, (100, 100))
            vec = resized.flatten().astype("float32")
            mu = float(vec.mean())
            sigma = float(vec.std()) + 1e-6
            vec = (vec - mu) / sigma
            norm = np.linalg.norm(vec) + 1e-8
            return vec / norm
        except Exception:
            return None

    def enroll_multi(self, face_gray_list):
        vecs = []
        for fg in face_gray_list:
            v = self._face_to_vec(fg)
            if v is not None:
                vecs.append(v)

        if len(vecs) < 8:
            return False

        ref = np.mean(np.stack(vecs, axis=0), axis=0)
        ref = ref / (np.linalg.norm(ref) + 1e-8)
        self.ref_vec = ref
        self._save_profile()
        return True

    def similarity(self, face_gray: np.ndarray):
        if self.ref_vec is None:
            return None
        vec = self._face_to_vec(face_gray)
        if vec is None:
            return None
        return float(np.dot(self.ref_vec, vec))


# In[17]:


#Telefon tespit sistemi (YOLOv5 + MediaPipe Hands)

class PhoneDetectionSystem:
    def __init__(self, use_yolo=True):
        self.use_yolo = use_yolo
        self.yolo_model = None

        if self.use_yolo:
            try:
                import torch
                print("[INFO] YOLOv5 modeli yukleniyor...")
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                self.yolo_model.conf = 0.4
                self.yolo_model.iou = 0.45
                self.yolo_model.eval()
                print("[OK] YOLOv5 modeli yuklendi")
            except Exception as e:
                print(f"[UYARI] YOLOv5 yuklenemedi: {e}")
                print("[INFO] Sadece el-tabanlı tespit kullanilacak")
                self.use_yolo = False

        try:
            if mp is None:
                raise ImportError("mediapipe mevcut degil")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[OK] MediaPipe Hands yuklendi")
        except Exception as e:
            print(f"[HATA] MediaPipe Hands yuklenemedi: {e}")
            self.hands = None

        self.phone_detected_frames = 0
        self.phone_state = False
        self.phone_detection_threshold = 8

        self.last_phone_alert_time = 0
        self.alert_cooldown = 5.0

        self.total_phone_detections = 0
        self.phone_usage_duration = 0.0
        self.phone_start_time = None

        self.detection_history = deque(maxlen=30)

    def detect_phone_yolo(self, frame):
        if not self.use_yolo or self.yolo_model is None:
            return False, []

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.yolo_model(rgb)
            detections = results.pandas().xyxy[0]

            if 'name' not in detections.columns:
                return False, []

            phone_detections = detections[detections['name'] == 'cell phone']
            if len(phone_detections) > 0:
                boxes = []
                for _, d in phone_detections.iterrows():
                    boxes.append([int(d['xmin']), int(d['ymin']), int(d['xmax']), int(d['ymax']), float(d['confidence'])])
                return True, boxes

            return False, []
        except Exception as e:
            print(f"[HATA] YOLO inference: {e}")
            return False, []

    def detect_phone_by_hands(self, frame, face_bbox=None):
        if self.hands is None:
            return False, []

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            if not results.multi_hand_landmarks:
                return False, []

            h, w = frame.shape[:2]
            suspicious_hands = []

            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)

                middle_tip = hand_landmarks.landmark[12]
                middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)

                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                hand_bbox = [min(xs), min(ys), max(xs), max(ys)]

                is_suspicious = False
                reasons = []

                if face_bbox is not None:
                    fx, fy, fw, fh = face_bbox
                    face_cy = fy + fh // 2

                    if wrist_x > fx + fw + 10 or wrist_x < fx - 10:
                        if fy - int(0.3 * fh) < wrist_y < fy + int(1.3 * fh):
                            is_suspicious = True
                            reasons.append("El yuzun yaninda")

                    if abs(wrist_y - face_cy) < fh * 0.8:
                        is_suspicious = True
                        reasons.append("El yuz seviyesinde")

                    if (fx - 100 < wrist_x < fx + fw + 100) and (fy - 100 < wrist_y < fy + fh + 100):
                        is_suspicious = True
                        reasons.append("El yuze cok yakin")
                else:
                    if wrist_y < h * 0.6:
                        is_suspicious = True
                        reasons.append("El ust kisimda")

                hand_angle = np.degrees(np.arctan2(middle_y - wrist_y, middle_x - wrist_x))
                if -45 < hand_angle < 45 or 135 < abs(hand_angle) < 180:
                    is_suspicious = True
                    reasons.append("Telefon tutus pozisyonu")

                if is_suspicious:
                    suspicious_hands.append({
                        "bbox": hand_bbox,
                        "reason": " + ".join(reasons),
                        "wrist": (wrist_x, wrist_y),
                        "confidence": 0.7
                    })

            return (len(suspicious_hands) > 0), suspicious_hands

        except Exception as e:
            print(f"[HATA] Hand detection: {e}")
            return False, []

    def update(self, frame, face_bbox=None):
        yolo_detected, yolo_boxes = self.detect_phone_yolo(frame)
        hand_detected, hand_infos = self.detect_phone_by_hands(frame, face_bbox)

        detected = yolo_detected or hand_detected
        self.detection_history.append(bool(detected))

        recent = sum(self.detection_history)
        ratio = recent / len(self.detection_history) if len(self.detection_history) else 0.0

        if detected:
            self.phone_detected_frames += 1
        else:
            self.phone_detected_frames = max(0, self.phone_detected_frames - 1)

        was_using = self.phone_state
        self.phone_state = (self.phone_detected_frames >= self.phone_detection_threshold) or (ratio >= 0.5)

        now = time.time()
        if self.phone_state:
            if self.phone_start_time is None:
                self.phone_start_time = now
                if not was_using:
                    self.total_phone_detections += 1
        else:
            if self.phone_start_time is not None:
                self.phone_usage_duration += now - self.phone_start_time
                self.phone_start_time = None

        should_alert = False
        if self.phone_state and (now - self.last_phone_alert_time > self.alert_cooldown):
            should_alert = True
            self.last_phone_alert_time = now

        info = {
            "yolo_detected": yolo_detected,
            "hand_detected": hand_detected,
            "yolo_boxes": yolo_boxes,
            "hand_infos": hand_infos,
            "confidence": ratio,
            "frames_detected": self.phone_detected_frames,
            "total_detections": self.total_phone_detections,
            "usage_duration": self.phone_usage_duration
        }
        return self.phone_state, info, should_alert

    def draw_detections(self, frame, detection_info):
        for box in detection_info["yolo_boxes"]:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"Telefon {conf:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for hi in detection_info["hand_infos"]:
            x1, y1, x2, y2 = hi["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(frame, hi["reason"], (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            wx, wy = hi["wrist"]
            cv2.circle(frame, (wx, wy), 5, (255, 0, 255), -1)

    def reset_statistics(self):
        self.total_phone_detections = 0
        self.phone_usage_duration = 0.0
        self.phone_start_time = None
        self.phone_detected_frames = 0
        self.phone_state = False
        self.detection_history.clear()


# In[18]:


# Sesli uyarı sistemi 

class AudioAlertSystem:
    def __init__(self):
        self.enabled = False
        self.last_alert_time = 0.0
        self.alert_cooldown = 3.0

        self.drowsy_sound = None
        self.distraction_sound = None
        self.warning_sound = None

        if pygame is None:
            return

        try:
            pygame.mixer.init()
            self.enabled = True
            self._create_sounds()
        except Exception as e:
            print(f"[UYARI] pygame ses sistemi baslatilamadi: {e}. Sesli uyarilar devre disi.")
            self.enabled = False

    def _create_sounds(self):
        if not self.enabled or pygame is None:
            return

        sample_rate = 22050

        def _tone(freq: float, duration: float):
            samples = int(sample_rate * duration)
            wave = np.sin(2 * np.pi * freq * np.linspace(0, duration, samples))
            wave = (wave * 32767).astype(np.int16)
            stereo = np.column_stack((wave, wave))
            return pygame.sndarray.make_sound(stereo)

        self.drowsy_sound = _tone(880, 0.5)
        self.distraction_sound = _tone(440, 0.5)
        self.warning_sound = _tone(220, 0.3)

    def play_alert(self, alert_type="drowsy"):
        if not self.enabled:
            return

        now = time.time()
        if now - self.last_alert_time < self.alert_cooldown:
            return

        snd = None
        if alert_type == "drowsy":
            snd = self.drowsy_sound
        elif alert_type == "distraction":
            snd = self.distraction_sound
        elif alert_type == "warning":
            snd = self.warning_sound

        if snd is not None:
            try:
                snd.play()
            except Exception as e:
                print(f"[UYARI] Ses calinamadi ({alert_type}): {e}. Ses devre disi.")
                self.enabled = False

        self.last_alert_time = now


# In[19]:


# Veri kayıt sistemi 

class DataLogger:
    def __init__(self, log_file=None):
        os.makedirs("sessions", exist_ok=True)

        if log_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join("sessions", f"surucu_oturumu_{ts}.json")
        else:
            if not os.path.isabs(log_file) and os.path.dirname(log_file) == "":
                log_file = os.path.join("sessions", log_file)

        self.log_file = log_file
        self.session_start = datetime.now()
        self.session_data = {
            "start_time": self.session_start.isoformat(),
            "end_time": None,
            "records": [],
            "statistics": {}
        }
        self.record_interval = 1.0
        self.last_record_time = 0

    def add_record(self, emotion_idx, emotion_labels, probs, eye_score,
                   drowsy_state, head_yaw, distracted, phone_using,
                   driver_similarity, driver_mismatch, mouth_open_score=None,
                   eye_gaze_distracted=False, eye_gaze_direction="center"):
        now = time.time()
        if now - self.last_record_time < self.record_interval:
            return

        record = {
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion_labels[emotion_idx],
            "emotion_probs": {
                "normal": float(probs[0]),
                "tired": float(probs[1])
            },
            "eye_openness": float(eye_score),
            "drowsy": (drowsy_state == "Drowsy"),
            "head_yaw": float(head_yaw),
            "distracted": bool(distracted),
            "phone_using": bool(phone_using),
            "driver_similarity": None if driver_similarity is None else float(driver_similarity),
            "driver_mismatch": bool(driver_mismatch),
            "eye_gaze_distracted": bool(eye_gaze_distracted),
            "eye_gaze_direction": str(eye_gaze_direction)
        }
        if mouth_open_score is not None:
            record["mouth_open_score"] = float(mouth_open_score)

        self.session_data["records"].append(record)
        self.last_record_time = now

    def save_session(self):
        self.session_data["end_time"] = datetime.now().isoformat()

        if self.session_data["records"]:
            rec = self.session_data["records"]
            self.session_data["statistics"] = {
                "total_duration_seconds": len(rec),
                "average_eye_openness": float(np.mean([r["eye_openness"] for r in rec])),
                "drowsy_percentage": float(sum([r["drowsy"] for r in rec]) / len(rec) * 100),
                "distracted_percentage": float(sum([r["distracted"] for r in rec]) / len(rec) * 100),
                "phone_using_percentage": float(sum([r["phone_using"] for r in rec]) / len(rec) * 100),
                "driver_mismatch_percentage": float(sum([r["driver_mismatch"] for r in rec]) / len(rec) * 100),
                "eye_gaze_distracted_percentage": float(sum([r.get("eye_gaze_distracted", False) for r in rec]) / len(rec) * 100),
                "emotion_distribution": {
                    "normal": float(sum([1 for r in rec if r["emotion"] == "Normal"]) / len(rec) * 100),
                    "tired": float(sum([1 for r in rec if r["emotion"] == "Yorgun"]) / len(rec) * 100),
                }
            }

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.session_data, f, indent=2, ensure_ascii=False)

        print(f"\n[LOG] Oturum kaydedildi: {self.log_file}")
        print(f"[LOG] Süre: {len(self.session_data['records'])} saniye")


# In[20]:


# Feature extractor (duygu özellikleri) 
class EmotionFeatureExtractor:
    def __init__(self):
        self.roi_coordinates = {
            'left_eye': (10, 15, 18, 10),
            'right_eye': (10, 30, 18, 10),
            'left_eyebrow': (5, 12, 8, 15),
            'right_eyebrow': (5, 28, 8, 15),
            'mouth': (32, 18, 12, 20),
        }

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        if image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.astype(np.uint8)

        features = []
        features.extend(self._extract_eye_features(img))
        features.extend(self._extract_eyebrow_features(img))
        features.extend(self._extract_mouth_features(img))
        features.extend(self._extract_global_features(img))
        return np.array(features, dtype=np.float32)

    def _extract_eye_features(self, img: np.ndarray):
        features = []
        for eye_name in ['left_eye', 'right_eye']:
            y, x, h, w = self.roi_coordinates[eye_name]
            eye_roi = img[y:y+h, x:x+w]
            if eye_roi.size == 0:
                features.extend([0.0, 0.0, 0.0, 0.0])
                continue

            eye_thresh = cv2.adaptiveThreshold(
                eye_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 2
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            eye_morph = cv2.morphologyEx(eye_thresh, cv2.MORPH_OPEN, kernel)

            features.append(np.sum(eye_morph == 255) / eye_morph.size)
            features.append(np.mean(eye_roi) / 255.0)
            features.append(np.std(eye_roi) / 255.0)

            contours, _ = cv2.findContours(eye_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            features.append(len(contours) / 10.0)
        return features

    def _extract_eyebrow_features(self, img: np.ndarray):
        features = []
        for brow_name in ['left_eyebrow', 'right_eyebrow']:
            y, x, h, w = self.roi_coordinates[brow_name]
            brow_roi = img[y:y+h, x:x+w]
            if brow_roi.size == 0:
                features.extend([0.0, 0.0, 0.0])
                continue

            _, brow_thresh = cv2.threshold(brow_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            features.append(np.sum(brow_thresh == 255) / brow_thresh.size)

            upper_half = np.mean(brow_roi[:h//2, :])
            lower_half = np.mean(brow_roi[h//2:, :])
            features.append((upper_half - lower_half) / 255.0)

            sobelx = cv2.Sobel(brow_roi, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(brow_roi, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.sqrt(sobelx**2 + sobely**2)
            features.append(np.mean(mag) / 255.0)
        return features

    def _extract_mouth_features(self, img: np.ndarray):
        features = []
        y, x, h, w = self.roi_coordinates['mouth']
        mouth_roi = img[y:y+h, x:x+w]
        if mouth_roi.size == 0:
            features.extend([0.0] * 6)
            return features

        mouth_thresh = cv2.adaptiveThreshold(
            mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mouth_closed = cv2.morphologyEx(mouth_thresh, cv2.MORPH_CLOSE, kernel)

        vertical_profile = np.sum(mouth_closed, axis=1)
        features.append(np.max(vertical_profile) / (w * 255))

        horizontal_profile = np.sum(mouth_closed, axis=0)
        features.append(np.sum(horizontal_profile > 0) / w)

        features.append(np.mean(mouth_roi[:h//2, :]) / 255.0)
        features.append(np.mean(mouth_roi[h//2:, :]) / 255.0)

        left_half = np.mean(mouth_roi[:, :w//2])
        right_half = np.mean(mouth_roi[:, w//2:])
        features.append(abs(left_half - right_half) / 255.0)

        contours, _ = cv2.findContours(mouth_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.append(len(contours) / 5.0)
        return features

    def _extract_global_features(self, img: np.ndarray):
        features = []
        features.append(np.mean(img) / 255.0)
        features.append(np.std(img) / 255.0)

        hist = cv2.calcHist([img], [0], None, [16], [0, 256]).flatten()
        hist_sum = hist.sum() if hist.sum() > 0 else 1.0
        hist = hist / hist_sum

        features.append(np.mean(hist[:8]))
        features.append(np.mean(hist[8:]))

        left_half = img[:, :img.shape[1]//2]
        right_half = cv2.flip(img[:, img.shape[1]//2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        if min_width > 0:
            diff = np.abs(left_half[:, :min_width] - right_half[:, :min_width])
            features.append(np.mean(diff) / 255.0)
        else:
            features.append(0.0)

        gabor_kernel = cv2.getGaborKernel((5, 5), 3, 0, 10, 0.5, 0)
        gabor_filtered = cv2.filter2D(img, cv2.CV_64F, gabor_kernel)
        features.append(np.mean(np.abs(gabor_filtered)) / 255.0)
        return features


class SimpleNNWrapper:
    def __init__(self, model_path: str):
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.weights = model_data['weights']
        self.biases = model_data['biases']
        self.activation_type = model_data['activation_type']

    def _activation(self, z):
        if self.activation_type == 'relu':
            return np.maximum(0, z)
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(a, w) + b
            if i == len(self.weights) - 1:
                a = self._softmax(z)
            else:
                a = self._activation(z)
        return a


# In[21]:


#Göz yönü (iris tracking) + göz açıklığı + kafa pozu

class EyeGazeDetector:
    def __init__(self, gaze_threshold=0.15, hold_seconds=2.5, release_seconds=0.6):
        self.gaze_threshold = float(gaze_threshold)
        self.hold_seconds = float(hold_seconds)
        self.release_seconds = float(release_seconds)

        self._left_gaze_since = None
        self._right_gaze_since = None
        self._center_since = None
        self.state = False
        self.last_direction = "center"

    def _get_eye_gaze_direction(self, face_landmarks):
        try:
            left_iris = face_landmarks[468]
            left_corner_left = face_landmarks[33]
            left_corner_right = face_landmarks[133]

            right_iris = face_landmarks[473]
            right_corner_left = face_landmarks[362]
            right_corner_right = face_landmarks[263]

            left_eye_width = abs(left_corner_right.x - left_corner_left.x)
            left_iris_offset = (left_iris.x - left_corner_left.x) / (left_eye_width + 1e-6)

            right_eye_width = abs(right_corner_right.x - right_corner_left.x)
            right_iris_offset = (right_iris.x - right_corner_left.x) / (right_eye_width + 1e-6)

            avg_offset = (left_iris_offset + right_iris_offset) / 2.0

            if avg_offset < (0.5 - self.gaze_threshold):
                return "left", avg_offset
            elif avg_offset > (0.5 + self.gaze_threshold):
                return "right", avg_offset
            else:
                return "center", avg_offset

        except Exception:
            return "center", 0.5

    def update(self, face_landmarks):
        direction, offset = self._get_eye_gaze_direction(face_landmarks)
        self.last_direction = direction
        now = time.time()

        if direction in ("left", "right"):
            self._center_since = None

            if direction == "left":
                if self._left_gaze_since is None:
                    self._left_gaze_since = now
                    self._right_gaze_since = None
                if (now - self._left_gaze_since) >= self.hold_seconds:
                    self.state = True

            if direction == "right":
                if self._right_gaze_since is None:
                    self._right_gaze_since = now
                    self._left_gaze_since = None
                if (now - self._right_gaze_since) >= self.hold_seconds:
                    self.state = True

        else:
            self._left_gaze_since = None
            self._right_gaze_since = None

            if self._center_since is None:
                self._center_since = now

            if self.state and (now - self._center_since) >= self.release_seconds:
                self.state = False

        return self.state, direction, offset


def _lm_to_np(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)

def compute_eye_openness(face_landmarks):
    left_corner_outer = _lm_to_np(face_landmarks, 33)
    left_corner_inner = _lm_to_np(face_landmarks, 133)
    left_upper = (_lm_to_np(face_landmarks, 159) + _lm_to_np(face_landmarks, 160)) / 2.0
    left_lower = (_lm_to_np(face_landmarks, 145) + _lm_to_np(face_landmarks, 144)) / 2.0

    right_corner_outer = _lm_to_np(face_landmarks, 263)
    right_corner_inner = _lm_to_np(face_landmarks, 362)
    right_upper = (_lm_to_np(face_landmarks, 386) + _lm_to_np(face_landmarks, 387)) / 2.0
    right_lower = (_lm_to_np(face_landmarks, 374) + _lm_to_np(face_landmarks, 373)) / 2.0

    left_vert = np.linalg.norm(left_upper - left_lower)
    left_horiz = np.linalg.norm(left_corner_outer - left_corner_inner) + 1e-6
    left_ratio = left_vert / left_horiz

    right_vert = np.linalg.norm(right_upper - right_lower)
    right_horiz = np.linalg.norm(right_corner_outer - right_corner_inner) + 1e-6
    right_ratio = right_vert / right_horiz

    eye_ratio = (left_ratio + right_ratio) / 2.0

    min_ratio = 0.15
    max_ratio = 0.35
    norm = (eye_ratio - min_ratio) / (max_ratio - min_ratio)
    eye_open_score = float(np.clip(norm, 0.0, 1.0))
    return eye_open_score, eye_ratio

def compute_head_pose(face_landmarks, img_width, img_height):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    image_points = np.array([
        (face_landmarks[1].x * img_width, face_landmarks[1].y * img_height),
        (face_landmarks[152].x * img_width, face_landmarks[152].y * img_height),
        (face_landmarks[33].x * img_width, face_landmarks[33].y * img_height),
        (face_landmarks[263].x * img_width, face_landmarks[263].y * img_height),
        (face_landmarks[61].x * img_width, face_landmarks[61].y * img_height),
        (face_landmarks[291].x * img_width, face_landmarks[291].y * img_height)
    ], dtype=np.float64)

    focal_length = img_width
    center = (img_width / 2, img_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    return yaw, pitch, roll


# In[22]:


# Uyku (göz kapalı) + dikkat dağınıklığı (yaw) state makinesi

class DriverSafetySystem:
    def __init__(self, eye_open_threshold=0.30, drowsy_frames_threshold=18, smoothing_window=8):
        self.eye_open_threshold = eye_open_threshold
        self.drowsy_frames_threshold = drowsy_frames_threshold
        self.eye_score_buffer = deque(maxlen=smoothing_window)
        self.closed_eye_frames = 0
        self.state = "Alert"

    def update(self, raw_eye_open_score):
        self.eye_score_buffer.append(raw_eye_open_score)
        smooth_eye_score = float(np.mean(self.eye_score_buffer))

        eye_is_closed = smooth_eye_score < self.eye_open_threshold
        if eye_is_closed:
            self.closed_eye_frames += 1
        else:
            self.closed_eye_frames = 0

        self.state = "Drowsy" if self.closed_eye_frames >= self.drowsy_frames_threshold else "Alert"
        return smooth_eye_score, self.closed_eye_frames, self.state


class DistractionDetector:
    def __init__(self, yaw_threshold=25, hold_seconds=2.5, release_seconds=0.6):
        self.yaw_threshold = float(yaw_threshold)
        self.hold_seconds = float(hold_seconds)
        self.release_seconds = float(release_seconds)

        self._above_since = None
        self._below_since = None
        self.state = False

    def update(self, yaw_angle):
        now = time.time()
        yaw_abs = abs(float(yaw_angle))

        if yaw_abs > self.yaw_threshold:
            self._below_since = None
            if self._above_since is None:
                self._above_since = now
            if (now - self._above_since) >= self.hold_seconds:
                self.state = True
        else:
            self._above_since = None
            if self._below_since is None:
                self._below_since = now
            if self.state and (now - self._below_since) >= self.release_seconds:
                self.state = False

        return self.state


# In[23]:


#Ana sistem: FullDriverMonitoringSystem

class FullDriverMonitoringSystem:
    def __init__(self, model_path='emotion_model.pkl', norm_path='normalization_params.npz'):
        print("=" * 60)
        print("Tam Surucu Izleme Sistemi")
        print("=" * 60)

        self.model = None
        self.mean = None
        self.std = None
        try:
            self.model = SimpleNNWrapper(model_path)
            norm_data = np.load(norm_path)
            self.mean = norm_data['mean']
            self.std = norm_data['std']
            print("[OK] Duygu modeli yuklendi")
        except Exception as e:
            print(f"[UYARI] Model yuklenemedi: {e}, sadece kural tabani kullanilacak")

        self.emotion_labels = ['Normal', 'Yorgun']
        self.num_emotions = len(self.emotion_labels)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.feature_extractor = EmotionFeatureExtractor()

        self.last_emotion_idx = 0
        self.last_probabilities = np.array([0.95, 0.05], dtype=float)
        self.last_confidence = 0.95
        self.prob_history = deque(maxlen=10)

        self.fps_buffer = deque(maxlen=30)

        if mp is None:
            raise ImportError('mediapipe mevcut degil: FaceMesh baslatilamadi.')
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.safety_system = DriverSafetySystem(
            eye_open_threshold=0.20,
            drowsy_frames_threshold=25,
            smoothing_window=10
        )

        self.distraction_detector = DistractionDetector(
            yaw_threshold=25,
            hold_seconds=2.5,
            release_seconds=0.6
        )

        self.eye_gaze_detector = EyeGazeDetector(
            gaze_threshold=0.15,
            hold_seconds=2.5,
            release_seconds=0.6
        )
        print("[OK] Goz yonu tespit sistemi hazir")

        self.phone_detector = PhoneDetectionSystem(use_yolo=True)
        print("[OK] Telefon tespit sistemi hazir")

        try:
            self.audio_system = AudioAlertSystem()
            print("[OK] Ses sistemi baslatildi")
        except Exception as e:
            print(f"[UYARI] Ses sistemi baslatilamadi: {e}")
            self.audio_system = None

        self.data_logger = DataLogger()
        print("[OK] Veri kayit sistemi hazir")

        self.eye_score_history = deque(maxlen=150)

        self.driver_id = DriverIdentifier(profile_path="driver_profile.npy", sim_threshold=0.70)
        self.main_driver_enabled = (self.driver_id.ref_vec is not None)

        self.sim_history = deque(maxlen=60)
        self.sim_low_threshold = self.driver_id.sim_threshold
        self.sim_high_threshold = min(0.985, self.sim_low_threshold + 0.10)

        self.driver_mismatch_frames = 0
        self.driver_mismatch_threshold_frames = 60
        self.driver_mismatch_state = False

        self.driver_alert_cooldown = 10.0
        self.last_driver_alert_time = 0.0
        self.last_driver_similarity = None

        self.enroll_mode = False
        self.enroll_samples = []
        self.enroll_start_time = 0.0
        self.enroll_duration = 4.0
        self.enroll_blur_threshold = 70
        self.match_blur_threshold = 50

        self.frame_idx = 0
        self.emotion_every_n_frames = 2

        self.last_mouth_open_score = 0.0
        self.debug_scores = True

        self._eyes_closed_since = None
        self._last_emergency_sent = 0.0
        self._emergency_cooldown = EMERGENCY_COOLDOWN_SECONDS

    def _rule_based_emotion(self, smooth_eye_score, drowsy_state, head_down_score, mouth_open_score):
        mouth_yawn_threshold = 0.50
        head_down_thr = 0.12
        eye_closed_thr = 0.20

        mouth_open = float(mouth_open_score)
        head_down = (float(head_down_score) > head_down_thr)
        eyes_closed = (float(smooth_eye_score) < eye_closed_thr)

        if drowsy_state == "Drowsy":
            idx = 1
            probs = np.array([0.10, 0.90], dtype=float)
            conf = 0.90
            return idx, probs, conf

        yorgun_signals = 0
        if eyes_closed:
            yorgun_signals += 1
        if head_down:
            yorgun_signals += 1
        if mouth_open > mouth_yawn_threshold:
            yorgun_signals += 1

        if yorgun_signals >= 2:
            idx = 1
            probs = np.array([0.20, 0.80], dtype=float)
            conf = 0.80
            return idx, probs, conf

        idx = 0
        probs = np.array([0.95, 0.05], dtype=float)
        conf = 0.95
        return idx, probs, conf

    def predict_emotion_from_features(self, features: np.ndarray,
                                     smooth_eye_score, drowsy_state,
                                     head_down_score, mouth_open_score):
        rb_idx, rb_probs, rb_conf = self._rule_based_emotion(
            smooth_eye_score, drowsy_state, head_down_score, mouth_open_score
        )

        if self.model is None or self.mean is None or self.std is None:
            return rb_idx, rb_probs, rb_conf

        try:
            features_norm = (features - self.mean) / (self.std + 1e-8)
        except Exception:
            features_norm = features

        try:
            _ = self.model.predict_proba(features_norm.reshape(1, -1))[0]
        except Exception:
            pass

        return rb_idx, rb_probs, rb_conf

    def draw_eye_trend_graph(self, frame):
        if len(self.eye_score_history) < 2:
            return

        h, w = frame.shape[:2]
        graph_h = 80
        graph_w = 200
        graph_x = w - graph_w - 20
        graph_y = 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (graph_x-10, graph_y-10), (graph_x+graph_w+10, graph_y+graph_h+10), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "Goz Aciklik Trend", (graph_x, graph_y-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        pts = list(self.eye_score_history)
        step = graph_w / max(len(pts) - 1, 1)

        for i in range(len(pts) - 1):
            x1 = int(graph_x + i * step)
            y1 = int(graph_y + graph_h - (pts[i] * graph_h))
            x2 = int(graph_x + (i + 1) * step)
            y2 = int(graph_y + graph_h - (pts[i + 1] * graph_h))
            color_val = int(pts[i] * 255)
            color = (0, color_val, 255 - color_val)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)

        threshold_y = int(graph_y + graph_h - (0.30 * graph_h))
        cv2.line(frame, (graph_x, threshold_y), (graph_x + graph_w, threshold_y), (0, 0, 255), 1)


# In[24]:


#FullDriverMonitoringSystem devam: draw_overlay + run


def _fullsystem_draw_overlay_and_run_patch():
    # Bu fonksiyon sadece notebook düzeni için bir "patch" değil.
    # Aşağıda FullDriverMonitoringSystem sınıfına metodları ekleyeceğiz.
    pass


def _draw_overlay(self, frame, emotion_idx, probs, confidence,
                  smooth_eye_score, closed_eye_frames, drowsy_state,
                  fps, yaw_angle, distraction_state, phone_state,
                  sim_value, main_driver_enabled, mismatch_state,
                  enroll_mode, eye_gaze_distracted, eye_gaze_direction):

    h, w = frame.shape[:2]
    overlay = frame.copy()
    left_panel_w = 500 if self.debug_scores else 460
    left_panel_h = 370 if self.debug_scores else 340
    cv2.rectangle(overlay, (15, 15), (15 + left_panel_w, 15 + left_panel_h), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    cv2.putText(frame, "AKILLI SURUCU IZLEME", (25, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2)

    label = self.emotion_labels[emotion_idx]
    color = (40, 220, 120) if emotion_idx == 0 else (0, 170, 255)

    y = 70
    cv2.putText(frame, f"Duygu: {label}", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, f"Guven: {confidence*100:.0f}%", (25, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    y += 60
    cv2.putText(frame, f"Goz aciklik skoru: {smooth_eye_score:.2f}", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 80), 1)
    cv2.putText(frame, f"Art arda kapali kare: {closed_eye_frames}", (25, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)

    y += 55
    badge_color = (0, 0, 255) if drowsy_state == "Drowsy" else (0, 180, 70)
    badge_text = "UYKULU" if drowsy_state == "Drowsy" else "DIKKATLI"
    cv2.rectangle(frame, (25, y - 18), (25 + 120, y + 10), badge_color, -1)
    cv2.putText(frame, badge_text, (32, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    y += 40
    cv2.putText(frame, f"Bas Donusu (Yaw): {yaw_angle:.1f}", (25, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 255), 1)
    cv2.putText(frame, f"Dikkat Dag.: {'EVET' if distraction_state else 'HAYIR'}", (25, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 120), 1)

    gaze_color = (0, 255, 255) if eye_gaze_distracted else (200, 200, 200)
    gaze_text = f"Goz Yonu: {eye_gaze_direction.upper()}"
    if eye_gaze_distracted:
        gaze_text += " [UYARI]"
    cv2.putText(frame, gaze_text, (25, y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)

    cv2.putText(frame, f"Telefon: {'EVET' if phone_state else 'HAYIR'}", (25, y + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 255), 1)

    y += 95
    if enroll_mode:
        cv2.putText(frame, "ANA SURUCU KAYDI: ALINIYOR (SABIT DUR)", (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        if main_driver_enabled:
            sim_txt = "N/A" if sim_value is None else f"{sim_value:.3f}"
            cv2.putText(frame, f"Ana surucu: AKTIF (sim={sim_txt})", (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            cv2.putText(frame, "p: ana surucu guncelle   r: reset   q: cikis   t: test", (25, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        else:
            cv2.putText(frame, "Ana surucu: PASIF (p ile kaydet)", (25, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if self.debug_scores:
        y += 50
        cv2.putText(frame, f"DBG mouth_open_score: {self.last_mouth_open_score:.3f}", (25, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    cv2.rectangle(frame, (w - 110, 20), (w - 20, 46), (15, 120, 60), -1)
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 105, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    graph_x = w - 260
    graph_y = h - 120
    legend_overlay = frame.copy()
    cv2.rectangle(legend_overlay, (graph_x - 15, graph_y - 30), (graph_x + 220, graph_y + 90), (15, 15, 35), -1)
    cv2.addWeighted(legend_overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "Duygu Olasiliklari", (graph_x - 5, graph_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)

    for i, (lbl, prob) in enumerate(zip(self.emotion_labels, probs)):
        bar_len = int(float(prob) * 170)
        clr = (40, 200, 120) if i == 0 else (0, 170, 255)
        y0 = graph_y + i * 28
        cv2.rectangle(frame, (graph_x, y0), (graph_x + bar_len, y0 + 20), clr, -1)
        cv2.putText(frame, f"{lbl}: {prob*100:.0f}%", (graph_x + 5, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    self.draw_eye_trend_graph(frame)

    if drowsy_state == "Drowsy":
        alarm_overlay = frame.copy()
        cv2.rectangle(alarm_overlay, (0, 0), (w, 70), (0, 0, 180), -1)
        cv2.addWeighted(alarm_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "UYKU HALI ALARMI - GOZLER UZUN SUREDIR KAPALI", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    if distraction_state or eye_gaze_distracted:
        dis_overlay = frame.copy()
        cv2.rectangle(dis_overlay, (0, h - 70), (w, h), (0, 160, 255), -1)
        cv2.addWeighted(dis_overlay, 0.85, frame, 0.15, 0, frame)

        warning_text = "DIKKAT DAGINIKLIGI: LUTFEN YOLA ODAKLANIN"
        if eye_gaze_distracted:
            warning_text = "GOZLERINIZI YOLA DONDURUN!"

        cv2.putText(frame, warning_text, (25, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    if mismatch_state:
        drv_overlay = frame.copy()
        cv2.rectangle(drv_overlay, (0, 70), (w, 130), (0, 0, 255), -1)
        cv2.addWeighted(drv_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(frame, "FARKLI SURUCU ALGILANDI!", (25, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def _run(self, camera_id=0):
    print("=" * 60)
    print("KONTROLLER:")
    print("  q: Cikis ve oturumu kaydet")
    print("  p: Ana surucu olarak kaydet/guncelle (4.0 sn kayit)")
    print("=" * 60)

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[HATA] Kamera acilamadi!")
        return

    print("[OK] Sistem baslatildi, kamera hazir")

    while True:
        self.frame_idx += 1
        do_emotion = (self.frame_idx % self.emotion_every_n_frames == 0)

        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[HATA] Frame okunamadi, cikiliyor...")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)

        results = self.face_mesh.process(rgb)

        smooth_eye_score = 1.0
        closed_eye_frames = self.safety_system.closed_eye_frames
        drowsy_state = self.safety_system.state
        head_down_score = 0.0
        yaw_angle = 0.0
        distraction_state = False
        eye_gaze_distracted = False
        eye_gaze_direction = "center"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            raw_eye_open_score, _ = compute_eye_openness(face_landmarks)
            smooth_eye_score, closed_eye_frames, drowsy_state = self.safety_system.update(raw_eye_open_score)

            # --- 10 saniye göz kapalı -> WEB tetikle (web hata verirse telegram fallback) ---
            now = time.time()
            eyes_closed = (smooth_eye_score < self.safety_system.eye_open_threshold)

            if eyes_closed:
                if self._eyes_closed_since is None:
                    self._eyes_closed_since = now
                closed_sec = now - self._eyes_closed_since

                if (closed_sec >= EMERGENCY_EYES_CLOSED_SECONDS) and ((now - self._last_emergency_sent) >= self._emergency_cooldown):
                    session_file = getattr(self.data_logger, "log_file", None)
                    trigger_emergency_to_web_async(
                        reason=f"Gozler {int(EMERGENCY_EYES_CLOSED_SECONDS)} saniyedir kapali",
                        seconds=closed_sec,
                        session_filename=session_file,
                        extra={"type": "eyes_closed"}
                    )
                    self._last_emergency_sent = now
            else:
                self._eyes_closed_since = None

            h_img, w_img = frame.shape[:2]
            eye_indices = [33, 133, 159, 160, 145, 144, 263, 362, 386, 387, 374, 373]
            for idx in eye_indices:
                lm = face_landmarks[idx]
                cv2.circle(frame, (int(lm.x * w_img), int(lm.y * h_img)), 1, (0, 255, 255), -1)

            nose_lm = face_landmarks[1]
            left_eye_lm = face_landmarks[33]
            right_eye_lm = face_landmarks[263]
            eye_center_y = (left_eye_lm.y + right_eye_lm.y) / 2.0
            head_down_score = float(nose_lm.y - eye_center_y)

            yaw_angle, pitch, roll = compute_head_pose(face_landmarks, w_img, h_img)
            distraction_state = self.distraction_detector.update(yaw_angle)

            eye_gaze_distracted, eye_gaze_direction, _ = self.eye_gaze_detector.update(face_landmarks)

            if self.debug_scores:
                iris_left = face_landmarks[468]
                cv2.circle(frame, (int(iris_left.x * w_img), int(iris_left.y * h_img)), 2, (0, 255, 0), -1)
                iris_right = face_landmarks[473]
                cv2.circle(frame, (int(iris_right.x * w_img), int(iris_right.y * h_img)), 2, (0, 255, 0), -1)

        faces = self.face_cascade.detectMultiScale(
            gray_eq,
            scaleFactor=1.05,
            minNeighbors=7,
            minSize=(120, 120)
        )

        emotion_idx = self.last_emotion_idx
        probs = self.last_probabilities
        confidence = self.last_confidence

        face_bbox = None
        face_roi = None
        mouth_open_score = self.last_mouth_open_score

        if len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w_face, h_face = faces_sorted[0]
            face_bbox = (x, y, w_face, h_face)
            face_roi = frame[y:y + h_face, x:x + w_face]

            try:
                if do_emotion:
                    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (48, 48))
                    normalized_face = resized_face.astype('float32') / 255.0
                    features = self.feature_extractor.extract_features(normalized_face)

                    # mouth feature: features[14] doğru (mouth bloğunun ilk özelliği)
                    mouth_open_score = float(features[14]) if features.size > 14 else 0.0
                    self.last_mouth_open_score = mouth_open_score

                    emotion_idx, probs, confidence = self.predict_emotion_from_features(
                        features, smooth_eye_score, drowsy_state, head_down_score, mouth_open_score
                    )

                    self.last_emotion_idx = emotion_idx
                    self.last_probabilities = probs
                    self.last_confidence = confidence

                box_color = (0, 255, 0) if emotion_idx == 0 else (255, 100, 0)
                cv2.rectangle(frame, (x, y), (x + w_face, y + h_face), box_color, 3)

            except Exception as e:
                print(f"[HATA] predict_emotion: {e}")
        else:
            cv2.putText(frame, "YUZ ALGILANAMADI", (420, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # ENROLL MODE
        if self.enroll_mode and face_roi is not None:
            fg = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(fg, cv2.CV_64F).var()
            if blur_score > self.enroll_blur_threshold:
                self.enroll_samples.append(fg)

            if (time.time() - self.enroll_start_time) >= self.enroll_duration:
                ok = self.driver_id.enroll_multi(self.enroll_samples)
                self.enroll_mode = False
                if ok:
                    self.main_driver_enabled = True
                    self.sim_history.clear()
                    self.driver_mismatch_frames = 0
                    self.driver_mismatch_state = False
                    print("[OK] Ana surucu profili (coklu kare) kaydedildi/guncellendi.")
                else:
                    print("[UYARI] Ana surucu kaydi basarisiz (yeterli kaliteli kare yok).")

        # Telefon
        phone_state, phone_info, _ = self.phone_detector.update(frame, face_bbox)
        if phone_state:
            self.phone_detector.draw_detections(frame, phone_info)
            distraction_state = True

        if eye_gaze_distracted:
            distraction_state = True

        # ANA SÜRÜCÜ TANIMA
        driver_mismatch = False
        if self.main_driver_enabled and face_roi is not None and (not self.enroll_mode):
            try:
                fg = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(fg, cv2.CV_64F).var()
                if blur_score > self.match_blur_threshold:
                    sim = self.driver_id.similarity(fg)
                    if sim is not None:
                        self.sim_history.append(sim)

                sim_smooth = float(np.mean(self.sim_history)) if len(self.sim_history) else None
                self.last_driver_similarity = sim_smooth

                if sim_smooth is not None:
                    if not self.driver_mismatch_state:
                        if sim_smooth < self.sim_low_threshold:
                            self.driver_mismatch_frames += 1
                        else:
                            self.driver_mismatch_frames = max(0, self.driver_mismatch_frames - 5)

                        if self.driver_mismatch_frames >= self.driver_mismatch_threshold_frames:
                            self.driver_mismatch_state = True
                    else:
                        if sim_smooth > self.sim_high_threshold:
                            self.driver_mismatch_frames = max(0, self.driver_mismatch_frames - 6)
                        else:
                            self.driver_mismatch_frames += 1

                        if self.driver_mismatch_frames <= 5:
                            self.driver_mismatch_state = False

                driver_mismatch = self.driver_mismatch_state

                now = time.time()
                if driver_mismatch and (now - self.last_driver_alert_time > self.driver_alert_cooldown):
                    if self.audio_system:
                        self.audio_system.play_alert("distraction")
                    self.last_driver_alert_time = now

            except Exception as e:
                print(f"[HATA] driver identify: {e}")

        self.eye_score_history.append(smooth_eye_score)

        if self.audio_system:
            if drowsy_state == "Drowsy":
                self.audio_system.play_alert("drowsy")
            elif distraction_state:
                self.audio_system.play_alert("distraction")

        self.data_logger.add_record(
            emotion_idx, self.emotion_labels, probs,
            smooth_eye_score, drowsy_state, yaw_angle,
            distraction_state, phone_state,
            self.last_driver_similarity, driver_mismatch,
            mouth_open_score=mouth_open_score,
            eye_gaze_distracted=eye_gaze_distracted,
            eye_gaze_direction=eye_gaze_direction
        )

        fps = 1.0 / max(time.time() - start, 1e-6)
        self.fps_buffer.append(fps)
        avg_fps = float(np.mean(self.fps_buffer))

        self.draw_overlay(
            frame, emotion_idx, probs, confidence,
            smooth_eye_score, closed_eye_frames, drowsy_state,
            avg_fps, yaw_angle, distraction_state, phone_state,
            self.last_driver_similarity, self.main_driver_enabled,
            driver_mismatch, self.enroll_mode,
            eye_gaze_distracted, eye_gaze_direction
        )

        cv2.imshow("Tam Surucu Izleme Sistemi", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n[INFO] Cikis yapiliyor...")
            break
        elif key == ord("r"):
            print("\n[INFO] Istatistikler sifirlandi")
            self.phone_detector.reset_statistics()
            self.sim_history.clear()
            self.driver_mismatch_frames = 0
            self.driver_mismatch_state = False
        elif key == ord("p"):
            if face_roi is None:
                print("[UYARI] p: Yuz bulunamadi, ana surucu kaydedilemedi.")
            else:
                self.enroll_mode = True
                self.enroll_samples = []
                self.enroll_start_time = time.time()
                print("[INFO] Ana surucu kaydi basladi... (4.0 sn sabit dur)")
        elif key == ord("t"):
            session_file = getattr(self.data_logger, "log_file", None)
            print("[TEST] Manuel emergency tetikleniyor -> WEB (fallback telegram aktif)")
            trigger_emergency_to_web_async(
                reason="MANUEL TEST (driver icinden)",
                seconds=0.0,
                session_filename=session_file,
                extra={"type": "manual_test"}
            )

    print("\n[INFO] Oturum kaydediliyor...")
    self.data_logger.save_session()

    cap.release()
    cv2.destroyAllWindows()
    self.face_mesh.close()
    print("[OK] Sistem kapatildi")

# Metodları sınıfa bağla
FullDriverMonitoringSystem.draw_overlay = _draw_overlay
FullDriverMonitoringSystem.run = _run


# In[25]:


#Notebook'tan başlatma

def baslat_sistemi_konsoldan():
    system = FullDriverMonitoringSystem(
        model_path="emotion_model.pkl",
        norm_path="normalization_params.npz"
    )
    system.run(camera_id=0)


# In[26]:




# In[27]:




# In[ ]:




