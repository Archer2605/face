import cv2
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

# Hàm để huấn luyện nhận dạng khuôn mặt
def train_faces():
    # Đường dẫn tới thư mục chứa dữ liệu huấn luyện
    dataset_path = 'dataset'

    # Tạo đối tượng nhận dạng khuôn mặt
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Tạo đối tượng phát hiện khuôn mặt
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Hàm lấy ảnh khuôn mặt và nhãn từ thư mục dataset
    def get_images_and_labels(dataset_path):
        # Danh sách chứa ảnh khuôn mặt
        face_samples = []

        # Danh sách chứa nhãn
        labels = []

        # Duyệt qua các thư mục con trong thư mục dataset
        for root, dirs, files in os.walk(dataset_path):
            for dir_name in dirs:
                # Sử dụng tên thư mục làm nhãn
                label = dir_name

                # Đường dẫn tới thư mục con
                dir_path = os.path.join(root, dir_name)

                # Đường dẫn tới các ảnh trong thư mục con
                image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

                # Duyệt qua các ảnh trong thư mục con
                for image_path in image_paths:
                    # Mở ảnh và chuyển đổi sang ảnh grayscale
                    PIL_img = Image.open(image_path).convert('L')

                    # Chuyển đổi ảnh sang mảng numpy
                    img_numpy = np.array(PIL_img, 'uint8')

                    # Phát hiện khuôn mặt trong ảnh
                    faces = detector.detectMultiScale(img_numpy)

                    # Lặp qua các khuôn mặt phát hiện được
                    for (x, y, w, h) in faces:
                        # Thêm khuôn mặt vào danh sách face_samples
                        face_samples.append(img_numpy[y:y + h, x:x + w])

                        # Thêm nhãn vào danh sách labels
                        labels.append(label)

        return face_samples, labels

    print("\n[INFO] Đang huấn luyện khuôn mặt. Vui lòng đợi trong vài giây ...")
    faces, labels = get_images_and_labels(dataset_path)

    # Áp dụng Label Encoder để chuyển đổi các nhãn thành các số nguyên duy nhất
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    # Huấn luyện mô hình nhận dạng khuôn mặt
    recognizer.train(faces, np.array(labels))

    # Lưu mô hình đã huấn luyện
    recognizer.save('trainer/trainer.yml')

    # Ghi các nhãn cùng với thứ tự của chúng vào tệp labels.txt
    labels_file_path = 'labels.txt'
    with open(labels_file_path, 'w') as file:
        for index, label in enumerate(label_encoder.classes_):
            file.write(f'{label},{index+1}\n')

    print("\n[INFO] Đã huấn luyện {0} khuôn mặt. Kết thúc chương trình.".format(len(np.unique(labels))))

# Hàm để hiển thị các nhãn đã được huấn luyện
def display_trained_labels():
    # Đường dẫn tới tệp labels.txt
    labels_file_path = 'labels.txt'

    # Kiểm tra xem tệp labels.txt có tồn tại hay không
    if not os.path.isfile(labels_file_path):
        print("Tệp labels.txt không tồn tại.")
        return

    # Đọc các nhãn từ tệp labels.txt và hiển thị
    with open(labels_file_path, 'r') as file:
        lines = file.readlines()

        print("Các nhãn đã được huấn luyện:")
        for line in lines:
            label, index = line.strip().split(',')  # Tách nhãn và thứ tự từ dòng
            print(f'Nhãn: {label} - Thứ tự: {index}')  # In nhãn và thứ tự

# Gọi hàm để huấn luyện khuôn mặt và lưu các nhãn vào tệp labels.txt
train_faces()

# Gọi hàm để hiển thị các nhãn đã được huấn luyện
display_trained_labels()
