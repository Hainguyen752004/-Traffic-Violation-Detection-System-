# main.py — Multi-detector single-file with threads + mosaic
import cv2
import numpy as np
import threading
import time
import os
import json
from datetime import datetime
from ultralytics import YOLO

# ----------------- CONFIG -----------------
VIDEO_1 = "263846879_main_xxl.mp4"  # traffic light + line (doxanh)
VIDEO_2 = "xenguocchieu.mp4"        # reverse detection (nguocchieu)
VIDEO_3 = "274841890_main_xxl.mp4"  # zone detector (leole)
OUTPUT_DIR = "violations"
PANEL_W, PANEL_H = 480, 360   # display size per panel in mosaic
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------- Helpers -----------------
def append_json(path, record):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=4)
    with open(path, "r+", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except Exception:
            data = []
        data.append(record)
        f.seek(0); json.dump(data, f, indent=4, ensure_ascii=False); f.truncate()


# ----------------- Detector classes -----------------

class TrafficLightDetector:
    """Traffic light detector + line (doxanh)"""
    def __init__(self, video_path, name="TrafficLight"):
        self.name = name
        self.video_path = video_path
        self.model = YOLO("models/yolo11m.pt")
        self.light_model = YOLO("models/best_traffic_small_yolo.pt")
        self.cap = cv2.VideoCapture(video_path)
        self.json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_path))[0]}_violations.json")
        with open(self.json_path, "w", encoding="utf-8") as f: json.dump([], f, indent=4)

        self.line_p1 = [100, 400]
        self.line_p2 = [675, 365]
        if os.path.exists("line_doxanh.npy"):
            try:
                d = np.load("line_doxanh.npy", allow_pickle=True).item()
                self.line_p1, self.line_p2 = d["p1"], d["p2"]
            except Exception:
                pass

        # state
        self.trajectories = {}
        self.counted_ids = set()
        self.display_id_map = {}
        self.id_color_map = {}
        self.violation_count = 0
        self.non_violation_count = 0
        self.current_signal = "green"

        # thread-safe last_frame
        self.last_frame = None
        self.lock = threading.Lock()
        self.stopped = False

        # original frame size
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or PANEL_W
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or PANEL_H

    def handle_mouse(self, event, x, y, flags):
        """x,y are coordinates relative to original detector resolution"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if abs(x - self.line_p1[0]) < 15 and abs(y - self.line_p1[1]) < 15:
                self._drag = "p1"
            elif abs(x - self.line_p2[0]) < 15 and abs(y - self.line_p2[1]) < 15:
                self._drag = "p2"
            else:
                self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if getattr(self, "_drag", None) == "p1":
                self.line_p1 = [int(x), int(y)]
            elif getattr(self, "_drag", None) == "p2":
                self.line_p2 = [int(x), int(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _is_crossed(self, prev, curr):
        x1, y1 = self.line_p1; x2, y2 = self.line_p2
        a = y2 - y1; b = x1 - x2; c = x2*y1 - x1*y2
        def side(p): return a*p[0] + b*p[1] + c
        return side(prev) * side(curr) <= 0

    def _get_light_color(self, crop):
        if crop is None or crop.size == 0: return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_red = cv2.inRange(hsv, (0,70,50), (10,255,255))
        mask_green = cv2.inRange(hsv, (40,40,40), (90,255,255))
        mask_yellow = cv2.inRange(hsv, (15,100,100), (35,255,255))
        counts = {"red": cv2.countNonZero(mask_red),
                  "green": cv2.countNonZero(mask_green),
                  "yellow": cv2.countNonZero(mask_yellow)}
        return max(counts, key=counts.get) if max(counts.values()) > 50 else None

    def _log_violation(self, disp_id, label, status, filename=None):
        rec = {"id": disp_id, "label": label, "status": status,
               "time": time.strftime("%Y-%m-%d %H:%M:%S"), "image": filename}
        append_json(self.json_path, rec)

    def process_frame(self, frame):
        # detect traffic light
        try:
            lr = self.light_model(frame)
            if lr and lr[0].boxes is not None:
                for box in lr[0].boxes.xyxy.cpu().numpy():
                    x1,y1,x2,y2 = map(int, box)
                    col = self._get_light_color(frame[y1:y2, x1:x2])
                    if col:
                        self.current_signal = col
                        break
        except Exception:
            pass

        # detect vehicles
        try:
            results = self.model.track(frame, persist=True, tracker="botsort.yaml")
        except Exception:
            results = None

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            names = self.model.names

            for box, track_id, cls in zip(boxes, ids, clss):
                x1,y1,x2,y2 = map(int, box)
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                prev = self.trajectories.get(track_id, [])[-1] if track_id in self.trajectories else None
                self.trajectories.setdefault(track_id, []).append((cx,cy))
                label = names[int(cls)]

                if track_id not in self.counted_ids and prev is not None and self._is_crossed(prev, (cx,cy)):
                    disp_id = self.display_id_map.setdefault(track_id, len(self.display_id_map)+1)
                    status = "NON-VIOLATION"; color=(0,255,0); filename=None
                    if self.current_signal != "green":
                        status = "VIOLATION"
                        color = (0,0,255) if self.current_signal=="red" else (0,255,255)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size>0:
                            filename = os.path.join(OUTPUT_DIR, f"{label}_ID{disp_id}_{int(time.time())}.jpg")
                            cv2.imwrite(filename, crop)
                    if status=="VIOLATION": self.violation_count +=1
                    else: self.non_violation_count +=1
                    self.id_color_map[track_id] = color
                    self._log_violation(disp_id, label, status, filename)
                    self.counted_ids.add(track_id)

                if track_id in self.display_id_map:
                    disp_id = self.display_id_map[track_id]
                    color = self.id_color_map.get(track_id, (255,255,255))
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                    cv2.putText(frame, f"ID {disp_id} {label}", (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # draw line & small markers (NO panel)
        line_color = (0,255,0) if self.current_signal=="green" else ((0,255,255) if self.current_signal=="yellow" else (0,0,255))
        cv2.line(frame, tuple(self.line_p1), tuple(self.line_p2), line_color, 4)
        cv2.circle(frame, tuple(self.line_p1), 6, (255,255,255), -1)
        cv2.circle(frame, tuple(self.line_p2), 6, (255,255,255), -1)
        return frame

    def read_and_update(self):
        """Read a frame, process, and update last_frame (thread-safe)."""
        if self.stopped: return
        ret, frame = self.cap.read()
        if not ret:
            # loop back to start to avoid long blank flicker
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
        try:
            out = self.process_frame(frame)
        except Exception:
            out = frame
        with self.lock:
            self.last_frame = out

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass
        self.stopped = True


class ReverseDetector:
    """Nguoc chieu detector (line crossing, half-line red/green)"""
    def __init__(self, video_path, name="Reverse"):
        self.name = name
        self.video_path = video_path
        self.model = YOLO("models/yolo11m.pt")
        self.cap = cv2.VideoCapture(video_path)
        self.json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(video_path))[0]}_violations.json")
        with open(self.json_path, "w", encoding="utf-8") as f: json.dump([], f, indent=4)

        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or PANEL_W
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or PANEL_H

        self.line_p1 = [30, 350]
        self.line_p2 = [self.frame_w - 100, 350]

        self.trajectories = {}
        self.counted_ids = set()
        self.display_id_map = {}
        self.id_color_map = {}
        self.vehicle_status = {}
        self.violation_count = 0
        self.non_violation_count = 0

        self.last_frame = None
        self.lock = threading.Lock()
        self.stopped = False

    def handle_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            if abs(x - self.line_p1[0]) < 20 and abs(y - self.line_p1[1]) < 20:
                self._drag = "p1"
            elif abs(x - self.line_p2[0]) < 20 and abs(y - self.line_p2[1]) < 20:
                self._drag = "p2"
            else:
                self._drag = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if getattr(self, "_drag", None) == "p1":
                self.line_p1 = [int(x), int(y)]
            elif getattr(self, "_drag", None) == "p2":
                self.line_p2 = [int(x), int(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _is_crossed(self, prev, curr):
        x1, y1 = self.line_p1; x2, y2 = self.line_p2
        a = y2 - y1; b = x1 - x2; c = x2*y1 - x1*y2
        def side(p): return a*p[0] + b*p[1] + c
        return side(prev) * side(curr) <= 0

    def _section(self, prev, curr):
        x1,y1 = prev; x2,y2 = curr
        x3,y3 = self.line_p1; x4,y4 = self.line_p2
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-10: return 'non_violation'
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4))/denom
        ix = x1 + t*(x2-x1); iy = y1 + t*(y2-y1)
        line_len = ((x4-x3)**2 + (y4-y3)**2)**0.5
        if line_len == 0: return 'non_violation'
        pos = ((ix-x3)*(x4-x3) + (iy-y3)*(y4-y3)) / (line_len**2)
        return 'violation' if pos < 0.5 else 'non_violation'

    def process_frame(self, frame):
        try:
            results = self.model.track(frame, persist=True, tracker="botsort.yaml")
        except Exception:
            results = None

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            names = self.model.names

            for box, tid, cls in zip(boxes, ids, clss):
                x1,y1,x2,y2 = map(int, box)
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                prev = self.trajectories.get(tid, [])[-1] if tid in self.trajectories else None
                self.trajectories.setdefault(tid, []).append((cx,cy))
                label = names[int(cls)]

                if tid not in self.counted_ids and prev is not None and self._is_crossed(prev,(cx,cy)):
                    disp_id = self.display_id_map.setdefault(tid, len(self.display_id_map)+1)
                    section = self._section(prev,(cx,cy))
                    status = "VIOLATION" if section=='violation' else "NON-VIOLATION"
                    color = (0,0,255) if section=='violation' else (0,255,0)
                    if section == 'violation':
                        crop = frame[y1:y2, x1:x2]
                        if crop.size>0:
                            filename = os.path.join(OUTPUT_DIR, f"{label}_ID{disp_id}_violation_{int(time.time())}.jpg")
                            cv2.imwrite(filename, crop)
                            append_json(self.json_path, {"id": disp_id, "label": label, "status": "VIOLATION", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "image": filename})
                        self.violation_count += 1
                    else:
                        append_json(self.json_path, {"id": disp_id, "label": label, "status": "NON-VIOLATION", "time": time.strftime("%Y-%m-%d %H:%M:%S"), "image": None})
                        self.non_violation_count += 1
                    self.id_color_map[tid] = color
                    self.vehicle_status[tid] = status
                    self.counted_ids.add(tid)

                if tid in self.display_id_map:
                    disp_id = self.display_id_map[tid]
                    color = self.id_color_map.get(tid,(255,255,255))
                    status = self.vehicle_status.get(tid,"")
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                    cv2.putText(frame, f"ID {disp_id} {label}", (x1,y1-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.putText(frame, status, (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # draw half-red/half-green line
        mid = ((self.line_p1[0]+self.line_p2[0])//2, (self.line_p1[1]+self.line_p2[1])//2)
        cv2.line(frame, tuple(self.line_p1), mid, (0,0,255), 5)
        cv2.line(frame, mid, tuple(self.line_p2), (0,255,0), 5)
        cv2.circle(frame, tuple(self.line_p1), 6, (255,255,255), -1)
        cv2.circle(frame, tuple(self.line_p2), 6, (255,255,255), -1)
        return frame

    def read_and_update(self):
        if self.stopped: return
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return
        try:
            out = self.process_frame(frame)
        except Exception:
            out = frame
        with self.lock:
            self.last_frame = out

    def release(self):
        try: self.cap.release()
        except: pass
        self.stopped = True


class ZoneDetector:
    """Zone (leole) detector with polygon zone"""
    VEHICLE_CLASSES = [2,3,5,7]
    def __init__(self, source, name="Zone"):
        self.name = name
        self.source = source
        self.model = YOLO("models/yolo11m.pt")
        self.is_video = source.endswith(('.mp4','.avi','.mov','.mkv')) or isinstance(source,int)
        self.cap = cv2.VideoCapture(source) if self.is_video else None
        img = None
        if self.is_video:
            ok, img = self.cap.read()
            if not ok: img = None
        else:
            img = cv2.imread(source)
        self.clone = img.copy() if img is not None else None

        self.points = [(100,400),(200,350),(400,360),(500,420)]
        if os.path.exists("zone.npy"):
            try:
                pts = np.load("zone.npy", allow_pickle=True).tolist()
                self.points = [(int(p[0]), int(p[1])) for p in pts]
            except Exception:
                pass

        self.id_map = {}
        self.next_id = 1
        self.logged_ids = set()
        base_name = os.path.splitext(os.path.basename(source))[0] if isinstance(source,str) else "zone"
        self.json_path = os.path.join(OUTPUT_DIR, f"{base_name}_violations.json")
        if not os.path.exists(self.json_path):
            with open(self.json_path, "w", encoding="utf-8") as f: json.dump([], f, indent=4)

        self.last_frame = None
        self.lock = threading.Lock()
        self.stopped = False

    def handle_mouse(self, event, x, y, flags):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(self.points):
                dx = p[0] - x; dy = p[1] - y
                if dx*dx + dy*dy <= 15*15:
                    self._drag = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if getattr(self, "_drag", None) is not None:
                idx = self._drag
                self.points[idx] = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _point_in_poly(self, point, polygon):
        x, y = point; n = len(polygon); inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n+1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _bbox_in_zone(self, bbox):
        x1,y1,x2,y2 = bbox
        pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
        for p in pts:
            if self._point_in_poly(p, self.points): return True
        center = ((x1+x2)//2,(y1+y2)//2)
        return self._point_in_poly(center,self.points)

    def process_frame(self, frame):
        try:
            results = self.model.track(frame, persist=True, verbose=False, tracker="botsort.yaml")
        except Exception:
            results = None
        violations = []
        if results:
            for result in results:
                boxes = result.boxes
                if boxes is None: continue
                for box in boxes:
                    class_id = int(box.cls[0]); conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else -1
                    if class_id in self.VEHICLE_CLASSES and conf > 0.5 and track_id != -1:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        if self._bbox_in_zone([x1,y1,x2,y2]):
                            if track_id not in self.id_map:
                                self.id_map[track_id] = self.next_id; self.next_id += 1
                            if self.id_map[track_id] not in self.logged_ids:
                                append_json(self.json_path, {"id": self.id_map[track_id], "vehicle": self.model.names[class_id], "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "violation_type": "zone"})
                                self.logged_ids.add(self.id_map[track_id])
                            violations.append({'bbox':[x1,y1,x2,y2],'class_name': self.model.names[class_id], 'confidence': conf, 'custom_id': self.id_map[track_id]})
        # draw zone overlay & violations (no panel)
        overlay = frame.copy()
        if len(self.points) > 1:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(overlay, [pts], True, (0,255,0), 2)
            cv2.fillPoly(overlay, [pts], (0,255,0))
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        for p in self.points:
            cv2.circle(frame, p, 6, (0,0,255), -1)

        for v in violations:
            x1,y1,x2,y2 = v['bbox']
            lab = f"ID {v['custom_id']} | {v['class_name']} {v['confidence']:.2f}"
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            lsize = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1,y1-lsize[1]-8), (x1+lsize[0], y1), (0,0,255), -1)
            cv2.putText(frame, lab, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return frame

    def read_and_update(self):
        if self.stopped: return
        if self.is_video:
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return
        else:
            frame = self.clone.copy()
        try:
            out = self.process_frame(frame)
        except Exception:
            out = frame
        with self.lock:
            self.last_frame = out

    def release(self):
        try:
            if self.cap: self.cap.release()
        except:
            pass
        self.stopped = True


# ----------------- Thread loop: each detector updates its last_frame -----------------
def detector_thread_loop(detector, interval=0.03):
    while True:
        if getattr(detector, "stopped", False):
            break
        try:
            detector.read_and_update()
        except Exception:
            pass
        time.sleep(interval)


# ----------------- MAIN: start threads, show mosaic + panel, handle mouse -----------------
if __name__ == "__main__":
    # instantiate detectors
    d1 = TrafficLightDetector(VIDEO_1, name="TrafficLight")
    d2 = ReverseDetector(VIDEO_2, name="WrongWay")
    d3 = ZoneDetector(VIDEO_3, name="ZoneDetector")
    detectors = [d1, d2, d3]

    # start threads
    threads = []
    for det in detectors:
        t = threading.Thread(target=detector_thread_loop, args=(det,), daemon=True)
        t.start()
        threads.append(t)

    cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)

    # mapping mosaic pixel->panel index and rel coords
    def mosaic_to_panel(mx, my):
        col = mx // PANEL_W
        row = my // PANEL_H
        idx = int(row*2 + col)
        if idx < 0 or idx > 2:
            return None, None, None
        rel_x = mx - (col * PANEL_W)
        rel_y = my - (row * PANEL_H)
        return idx, rel_x, rel_y

    # mouse callback: scale rel coords to detector original resolution before forwarding
    def on_mouse(event, mx, my, flags, param):
        idx, rx, ry = mosaic_to_panel(mx, my)
        if idx is None: return
        det = detectors[idx]
        # scale to original detector coords
        if hasattr(det, "orig_w") and hasattr(det, "orig_h"):
            sx = det.orig_w / PANEL_W; sy = det.orig_h / PANEL_H
            ox = int(rx * sx); oy = int(ry * sy)
        else:
            # try fallback on frame dims
            if det.last_frame is not None:
                h, w = det.last_frame.shape[:2]
                ox = int(rx * (w / PANEL_W)); oy = int(ry * (h / PANEL_H))
            else:
                ox, oy = int(rx), int(ry)
        try:
            det.handle_mouse(event, ox, oy, flags)
        except Exception:
            pass

    cv2.setMouseCallback("Monitor", on_mouse)

    try:
        while True:
            # gather display frames (use last_frame if available)
            disp_frames = []
            for det in detectors:
                with getattr(det, "lock"):
                    last = getattr(det, "last_frame", None)
                if last is None:
                    # blank panel
                    disp = 255 * np.ones((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                else:
                    try:
                        disp = cv2.resize(last, (PANEL_W, PANEL_H))
                    except Exception:
                        disp = 255 * np.ones((PANEL_H, PANEL_W, 3), dtype=np.uint8)
                disp_frames.append(disp)

            # mosaic 2x2, bottom-right blank
            row1 = np.hstack([disp_frames[0], disp_frames[1]])
            row2 = np.hstack([disp_frames[2], 255*np.ones((PANEL_H, PANEL_W, 3), dtype=np.uint8)])
            mosaic = np.vstack([row1, row2])

            # draw summary panel below mosaic (height 100)
           # --- Dashboard Panel ---
            panel_h = 180   # cao hơn để không bị đè chữ
            panel = np.zeros((panel_h, mosaic.shape[1], 3), dtype=np.uint8)  # nền đen

            # Tiêu đề
            cv2.putText(panel, "TRAFFIC VIOLATION DASHBOARD", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

            # Camera 1
            cv2.putText(panel,
                        f"1: {d1.name} | Viol: {d1.violation_count} | NonViol: {d1.non_violation_count} | Signal: {d1.current_signal}",
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Camera 2
            cv2.putText(panel,
                        f"2: {d2.name} | Viol: {d2.violation_count} | NonViol: {d2.non_violation_count}",
                        (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Camera 3
            cv2.putText(panel,
                        f"3: {d3.name} | Total Viol: {len(d3.logged_ids)}",
                        (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Ghép panel xuống dưới mosaic
            final = np.vstack([mosaic, panel])

            cv2.imshow("Monitor", final)


            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                # save persistent configs
                try: np.save("line_doxanh.npy", {"p1": d1.line_p1, "p2": d1.line_p2})
                except: pass
                try: np.save("line_nguocchieu.npy", {"p1": d2.line_p1, "p2": d2.line_p2})
                except: pass
                try: np.save("zone.npy", np.array(d3.points))
                except: pass
                print("Saved line/zone.")
    finally:
        # cleanup
        for det in detectors:
            try: det.release()
            except: pass
        cv2.destroyAllWindows()
