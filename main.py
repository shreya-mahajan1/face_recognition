import os.path
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from test import test
import util
import face_recognition


class App:

    def __init__(self):
        self.cap = None
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.login_button_main_window = util.get_button(self.main_window, "login", "green", self.login)
        self.login_button_main_window.place(x=750, y=200)

        self.logout_button_main_window = util.get_button(self.main_window, 'logout', 'red', self.logout)
        self.logout_button_main_window.place(x=750, y=300)

        self.register_new_button_main_window = util.get_button(self.main_window, "register new user", "gray",
                                                               self.register_new_user, fg="black")
        self.register_new_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)
        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)
        self.log_path = './log.txt'

    def add_webcam(self, label):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise ValueError("Cannot open camera")

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()  # Capture frame-by-frame
        if ret:
            self.most_recent_capture_arr = frame  # Store the most recent frame
            img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            self.most_recent_capture_pil = Image.fromarray(img_)  # Convert to PIL Image
            imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)  # Convert PIL Image to ImageTk

            self._label.imgtk = imgtk  # Keep a reference to avoid garbage collection
            self._label.configure(image=imgtk)  # Update the label widget with the new image

        self._label.after(20, self.process_webcam)  # Repeat after 20 milliseconds

    def login(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir="C:/Users/mridu/PycharmProjects/pythonProject/pythonProject1/Silent-Face-Anti-Spoofing"
                          "-master/resources/anti_spoof_models",
                device_id=0
                )

        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},in\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def logout(self):

        label = test(
                image=self.most_recent_capture_arr,
                model_dir="C:/Users/mridu/PycharmProjects/pythonProject/pythonProject1/Silent-Face-Anti-Spoofing"
                          "-master/resources/anti_spoof_models",
                device_id=0
                )

        if label == 1:

            name = util.recognize(self.most_recent_capture_arr, self.db_dir)

            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
            else:
                util.msg_box('Hasta la vista !', 'Goodbye, {}.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{},out\n'.format(name, datetime.datetime.now()))
                    f.close()

        else:
            util.msg_box('Hey, you are a spoofer!', 'You are fake !')

    def register_new_user(self):
        self.register_new_user_window=tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, "Accept",
                                                                      "green", self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, "Try Again", "red",
                                                                      self.try_again_register_new_user)
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)
        self.entry_text_register_new_user=util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)
        self.text_label_register_new_user=util.get_text_label(self.register_new_user_window,"Please input username : ")
        self.text_label_register_new_user.place(x=750, y=70)
    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)
        util.msg_box('Success!', 'User was registered successfully !')
        self.register_new_user_window.destroy()
    def try_again_register_new_user(self):
        
        self.register_new_user_window.destroy()

    def start(self):
        self.main_window.mainloop()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture=self.most_recent_capture_arr.copy()


if __name__ == "__main__":
    app = App()
    app.start()

