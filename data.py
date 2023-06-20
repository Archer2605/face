import cv2
import os

cam = cv2.VideoCapture(0)

cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Nhập tên người dùng
user_name = input('\nNhập tên của bạn: ==> ')

# Tạo thư mục con với tên người dùng
user_folder = os.path.join('dataset', user_name)
os.makedirs(user_folder, exist_ok=True)

print("\nĐang tiến hành chụp khuôn mặt ...")
# Khởi tạo biến đếm số lượng khuôn mặt đã chụp
count = 0

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Lưu ảnh khuôn mặt vào thư mục con với tên người dùng
        img_path = os.path.join(user_folder, f"{str(count)}.jpg")
        cv2.imwrite(img_path, gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 50: # Chụp 50 khuôn mặt
         break

print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
