import tkinter as tk
import subprocess

class InterfaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ứng dụng Giao diện")
        self.geometry("400x300")

        # Label
        self.label = tk.Label(self, text="Chọn tệp để chạy:")
        self.label.pack(pady=10)

        # Buttons
        self.data_button = tk.Button(self, text="Chạy tệp Data", command=self.run_data_file)
        self.data_button.pack(pady=5)

        self.train_button = tk.Button(self, text="Chạy tệp Train", command=self.run_train_file)
        self.train_button.pack(pady=5)

        self.face_button = tk.Button(self, text="Chạy tệp Face", command=self.run_face_file)
        self.face_button.pack(pady=5)

    def run_data_file(self):
        subprocess.Popen(["python", "data.py"])

    def run_train_file(self):
        subprocess.Popen(["python", "train.py"])

    def run_face_file(self):
        subprocess.Popen(["python", "face.py"])

if __name__ == "__main__":
    app = InterfaceApp()
    app.mainloop()
