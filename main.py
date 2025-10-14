import cv2
import numpy as np
import time
import customtkinter as ctk
from PIL import Image

############################
# Konfigurasiýa sazlamalary #
############################

# Wideonyň ýoly
video_path = 'cars.mp4'

# Pikselden metre geçiriş koeffisiýenti (sazlamaly ýer)
pixel_to_meter = 0.05

# Kontur görnüş boýunça süzgüçler
MIN_CONTOUR_AREA = 1500
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20

# Yzarlaýyş sazlamalary
MAX_MATCH_DISTANCE_PX = 60
MAX_MISSING_FRAMES = 10
SPEED_SMOOTH_ALPHA = 0.3  # 0.0-1.0 aralygynda; ýokary bolsa täze bahasy köp agramly

# Reňkleme
TEXT_COLOR = (0, 255, 0)
DOT_COLOR = (0, 255, 0)

# Penjiräniň ölçegi (None bolsa üýtgetmeýär)
FRAME_RESIZE_WIDTH = None  # meselem: 1280

# Statik aktiwleri öňünden ýükleýäris
_raw_back_icon = cv2.imread('back_icon.png')
_BACK_ICON = None
if _raw_back_icon is not None:
    try:
        _BACK_ICON = cv2.resize(_raw_back_icon, (50, 50), interpolation=cv2.INTER_AREA)
    except Exception:
        _BACK_ICON = None

# Obýekt ID hasaplaýjy (her başlangyçda täzelenýär)
next_car_id = 1


def _open_video_capture(path: str):
    """Wideony açýar we durnukly FPS gaýtaryýar."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f'Wideony açyp bolmady: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        fps = 30.0
    return cap, float(fps)


def _build_background_subtractor():
    """Gowy netijeler üçin MOG2 sazlamalary bilen BG subtractor döret."""
    # detectShadows=True – kölege 127 hökmünde bellener, soňra eşikleme bilen aýyrarys
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)

def _preprocess_frame_for_mask(frame_bgr: np.ndarray, bg_subtractor) -> np.ndarray:
    """
    Kadry öňünden gauss ýüwürme, fondan aýyrma, kölegeleri aýyrmak,
    soň morfologiýa amallary bilen arassala.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    raw_mask = bg_subtractor.apply(blurred)

    # Kölegeleri aýyrmak üçin eşikleme (MOG2-de: 0-bg, 127-kölege, 255-foreground)
    _, fgmask = cv2.threshold(raw_mask, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # ownuk şowhunlary aýyrmak
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    # gözenekleri doldurmak we bitewilik
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fgmask

def start_detection():
    """Wideony oka, obýektleri kesgitläp tizligini hasapla we görkez."""
    global next_car_id

    # Her başlangyçda arassalamak
    next_car_id = 1
    car_tracks = {}

    # Penjire taýýarlamak
    cv2.namedWindow('Vehicle Speed Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Vehicle Speed Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Wideony açmak we BG subtractor döretmek
    try:
        cap, nominal_fps = _open_video_capture(video_path)
    except Exception as e:
        print(str(e))
        cv2.destroyAllWindows()
        return

    bg_subtractor = _build_background_subtractor()

    prev_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Islendik ýagdaýda ululygy üýtgetmek
        if FRAME_RESIZE_WIDTH is not None and FRAME_RESIZE_WIDTH > 0:
            h, w = frame.shape[:2]
            scale = FRAME_RESIZE_WIDTH / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        now = time.perf_counter()
        # Runtime FPS-dan peýdalanyp, gaty kiçi dt-ni çäklendirýäris
        dt = max(now - prev_time, 1.0 / max(nominal_fps, 1.0) / 4.0)
        prev_time = now

        # Maskany taýýarlamak
        fgmask = _preprocess_frame_for_mask(frame, bg_subtractor)

        # Konturlary almak diňe daşky (external)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []  # (center_xy, bbox)
        for contour in contours:
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < MIN_BOX_WIDTH or h < MIN_BOX_HEIGHT:
                continue
            center = (int(x + w / 2), int(y + h / 2))
            detections.append((center, (x, y, w, h)))

        detected_centers = [c for c, _ in detections]

        # Greedy many-to-one jübütleşdirme
        unmatched_track_ids = set(car_tracks.keys())
        unmatched_detection_idxs = set(range(len(detected_centers)))

        pair_candidates = []  # (dist, track_id, det_idx)
        for track_id, track in car_tracks.items():
            if track.get('position') is None:
                continue
            track_pos = np.array(track['position'], dtype=np.float32)
            for det_idx, det_center in enumerate(detected_centers):
                det_pos = np.array(det_center, dtype=np.float32)
                dist = float(np.linalg.norm(det_pos - track_pos))
                pair_candidates.append((dist, track_id, det_idx))

        pair_candidates.sort(key=lambda t: t[0])

        # Üstünlikli jübütleşdirmeler boýunça täzelenýär
        for dist, track_id, det_idx in pair_candidates:
            if dist > MAX_MATCH_DISTANCE_PX:
                break
            if track_id not in unmatched_track_ids or det_idx not in unmatched_detection_idxs:
                continue

            det_center, det_bbox = detections[det_idx]
            prev_pos = car_tracks[track_id]['position']
            car_tracks[track_id]['position'] = det_center
            car_tracks[track_id]['bbox'] = det_bbox
            car_tracks[track_id]['frames_since_seen'] = 0

            if prev_pos is not None:
                pixel_distance = float(np.linalg.norm(np.array(det_center, dtype=np.float32) - np.array(prev_pos, dtype=np.float32)))
                # Saýlama görnüşi: wagtlaýyn FPS bilen tizligi hasapla (km/h)
                instant_speed = (pixel_distance * pixel_to_meter) / dt * 3.6
            else:
                instant_speed = 0.0

            prev_smooth = car_tracks[track_id].get('speed_kmh', 0.0)
            smooth_speed = SPEED_SMOOTH_ALPHA * instant_speed + (1.0 - SPEED_SMOOTH_ALPHA) * prev_smooth
            car_tracks[track_id]['speed_kmh'] = smooth_speed

            unmatched_track_ids.discard(track_id)
            unmatched_detection_idxs.discard(det_idx)

        # Täze kesgitlenenler üçin täze yzarlama döredilýär
        for det_idx in list(unmatched_detection_idxs):
            det_center, det_bbox = detections[det_idx]
            car_tracks[next_car_id] = {
                'position': det_center,
                'bbox': det_bbox,
                'speed_kmh': 0.0,
                'frames_since_seen': 0,
            }
            next_car_id += 1

        # Ýitip giden yzarlamalary arassala
        for track_id in list(unmatched_track_ids):
            car_tracks[track_id]['frames_since_seen'] = car_tracks[track_id].get('frames_since_seen', 0) + 1
            if car_tracks[track_id]['frames_since_seen'] > MAX_MISSING_FRAMES:
                del car_tracks[track_id]

        # Görkezmek
        for track in car_tracks.values():
            if track.get('position') is None:
                continue
            cx, cy = track['position']
            speed = float(track.get('speed_kmh', 0.0))
            if speed > 0.1:
                cv2.putText(
                    frame,
                    f'{speed:.2f} km/h',
                    (int(cx), int(cy) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    TEXT_COLOR,
                    2,
                )
            cv2.circle(frame, (int(cx), int(cy)), 5, DOT_COLOR, -1)

        # "Back" düwmesini goý
        if _BACK_ICON is not None:
            h, w = _BACK_ICON.shape[:2]
            frame[10:10 + h, 10:10 + w] = _BACK_ICON

        cv2.imshow('Vehicle Speed Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # ESC – yzyna
            break

    cap.release()
    cv2.destroyAllWindows()

def main() -> None:
    # CustomTkinter UI döredýäris
    app = ctk.CTk()
    app.title('Vehicle Speed Detection')
    app.geometry(f'{app.winfo_screenwidth()}x{app.winfo_screenheight()}')

    # Logo goşýarys (bar bolsa)
    _logo_image = None
    try:
        _pil_logo = Image.open('titu.png')
        _logo_image = ctk.CTkImage(_pil_logo, size=(200, 200))
    except Exception:
        _logo_image = None

    if _logo_image is not None:
        logo_label = ctk.CTkLabel(app, image=_logo_image, text=None)
        logo_label.pack(pady=20)

    label = ctk.CTkLabel(app, text='Masynlaryň tizligini barlamak', font=('Arial', 24))
    label.pack(pady=40)

    start_button = ctk.CTkButton(app, text='Başlat', command=start_detection, font=('Arial', 18))
    start_button.pack(pady=20)

    quit_button = ctk.CTkButton(app, text='Çyk', command=app.quit, font=('Arial', 18))
    quit_button.pack(pady=20)

    # "All right reserved" textini goşýarys
    footer_label = ctk.CTkLabel(app, text='©2025 All rights reserved', font=('Arial', 14))
    footer_label.pack(side='bottom', pady=10)

    app.mainloop()


if __name__ == '__main__':
    main()
