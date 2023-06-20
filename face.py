# Import các thư viện cần thiết
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Hàm để nhận dạng khuôn mặt
def recognize_faces():
    # Tạo một đối tượng nhận dạng khuôn mặt với LBPH
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Đọc mô hình đã được huấn luyện từ tập tin
    recognizer.read('trainer/trainer.yml')

    # Đường dẫn đến tập tin chứa haarcascade (dùng để phát hiện khuôn mặt)
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Chọn font chữ để vẽ text trên ảnh
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Tạo đối tượng LabelEncoder để chuyển đổi nhãn
    label_encoder = LabelEncoder()

    # Đọc tệp tin chứa tên người dùng và nhãn tương ứng
    with open('labels.txt', 'r') as file:
        lines = file.readlines()
        # Lấy danh sách tên người dùng
        names = [line.strip().split(',')[0] for line in lines]
        # Lấy danh sách nhãn
        labels = [line.strip().split(',')[1] for line in lines]
        # Mã hóa nhãn thành số nguyên
        encoded_labels = label_encoder.fit_transform(labels)

    # Khởi tạo và bắt đầu capture video thời gian thực từ webcam
    cam = cv2.VideoCapture(0)
    # cam = cv2.VideoCapture("1.mp4")

    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    # Vòng lặp vô tận để liên tục nhận dạng khuôn mặt trong video
    while True:
        # Đọc một frame từ video
        ret, img = cam.read()
        # Chuyển frame sang ảnh grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt trong ảnh
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

        # Với mỗi khuôn mặt phát hiện được
        for (x, y, w, h) in faces:
            # Vẽ hình chữ nhật xung quanh khuôn mặt
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Lấy phần ảnh chứa khuôn mặt
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            # Dự đoán nhãn khuôn mặt
            label_id, confidence = recognizer.predict(roi_gray)

            # Chuyển đổi nhãn về tên người dùng
            name = names[label_id]

            # Hiển thị tên và độ tin cậy lên ảnh
            if confidence < 100:
                text = f"{name} ({confidence:.2f}%)"
            else:
                text = "Unknown"

            cv2.putText(img, text, (x, y - 10), font, 0.9, (0, 255, 0), 2)

        # Hiển thị frame đã được nhận dạng khuôn mặt
        cv2.imshow('camera', img)

        # Nếu nhấn 'q', thoát vòng lặp
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Khi thoát vòng lặp, giải phóng camera và đóng cửa sổ
    cam.release()
    cv2.destroyAllWindows()

# Gọi hàm nhận dạng khuôn mặt
recognize_faces()
