import cv2
import numpy as np
from collections import defaultdict
import customtkinter as ctk
from PIL import Image

# Video açýarys
video_path = 'cars.mp4'
cap = cv2.VideoCapture(video_path)

# FPS-i dogry almak
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30

# Obýektleri kesgitlemek üçin fon arassalaýjy
fgbg = cv2.createBackgroundSubtractorMOG2()

# Maşynlaryň ýerleşişini we tizligini saklaýarys
car_tracks = defaultdict(lambda: {'position': None, 'speed': 0, 'frames': 0})

# Pikselden metre geçiriş koeffisiýenti (sazlamaly)
pixel_to_meter = 0.05

# Kadrlaryň arasyndaky wagt
frame_time = 1 / fps

# Obýekt ID hasaplaýjy
next_car_id = 1

# Wideony ýapmak üçin bellik
def stop_video():
    cap.release()
    cv2.destroyAllWindows()

# CustomTkinter interfeýsi
def start_detection():
    global next_car_id
    cv2.namedWindow('Vehicle Speed Detection', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Vehicle Speed Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Çal reňk we fon arassalaýyş
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)

        # Konturlary tapmak
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        detected_centers = []

        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                continue

            (x, y, w, h) = cv2.boundingRect(contour)

            # Maşynyň merkezini tapýarys
            center = (int(x + w / 2), int(y + h / 2))
            detected_centers.append(center)

            # Bar bolan maşynlary yzarlamak
            found = False
            for car_id, track in car_tracks.items():
                if track['position'] is not None:
                    distance = np.linalg.norm(np.array(center) - np.array(track['position']))
                    if distance < 50:
                        speed = (distance * pixel_to_meter) / frame_time * 3.6
                        car_tracks[car_id]['position'] = center
                        car_tracks[car_id]['speed'] = speed
                        car_tracks[car_id]['frames'] = 0
                        found = True
                        break

            if not found:
                car_tracks[next_car_id] = {'position': center, 'speed': 0, 'frames': 0}
                next_car_id += 1

        # Ýitip giden maşynlary aýyrmak
        for car_id in list(car_tracks):
            car_tracks[car_id]['frames'] += 1
            if car_tracks[car_id]['frames'] > 10:
                del car_tracks[car_id]

        # Maşynlary we tizligini görkezmek
        for car_id, track in car_tracks.items():
            if track['position'] is not None:
                (cx, cy) = track['position']
                speed = track['speed']
                if speed > 0:
                    cv2.putText(frame, f'{speed:.2f} km/h', (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # "Back" düwmesi
        back_button = cv2.imread('back_icon.png')
        if back_button is not None:
            frame[10:60, 10:60] = cv2.resize(back_button, (50, 50))

        cv2.imshow('Vehicle Speed Detection', frame)

        key = cv2.waitKey(30)
        if key == ord('q'):
            break
        elif key == 27:  # ESC düwmesi bilen "Back" funksiýasy
            stop_video()
            return

    stop_video()

# CustomTkinter UI döredýäris
app = ctk.CTk()
app.title('Vehicle Speed Detection')
app.geometry(f'{app.winfo_screenwidth()}x{app.winfo_screenheight()}')

# Logo goşýarys
logo_image = ctk.CTkImage(Image.open('titu.png'), size=(200, 200))
logo_label = ctk.CTkLabel(app, image=logo_image, text=None)
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
