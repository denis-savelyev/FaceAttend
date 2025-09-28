import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from attendance import AttendanceManager
from face_db import FaceDatabase
from gui import AppUI


class FaceRecognitionApp:
    def __init__(self):
        # UI root
        self.main_window = tk.Tk()

        # Face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Database and attendance managers
        self.db = FaceDatabase()
        self.db.load()
        self.attendance = AttendanceManager()

        # Runtime state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_detection: Optional[str] = None
        self.recognition_threshold = 0.6  # 0-1, higher = stricter
        self.paused = False

        # Build UI via AppUI and start camera
        self.ui = AppUI(
            self.main_window,
            on_confirm=self.confirm_identity,
            on_reject=self.reject_identity,
            on_add_person=self.add_new_person,
            on_retrain=self.retrain_model,
            on_view_db=self.view_database,
            on_clear_db=self.clear_database,
            on_export_attendance=self.export_attendance_csv,
            on_threshold_change=self.update_threshold,
            initial_threshold=self.recognition_threshold,
        )
        self.start_camera()

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera!")
                return

            self.is_running = True
            self.camera_thread = threading.Thread(target=self.update_camera, daemon=True)
            self.camera_thread.start()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")

    def update_camera(self):
        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    if not self.paused:
                        self.process_faces(frame, gray, faces)
                else:
                    self.current_detection = None
                    if not self.paused:
                        self.main_window.after(0, lambda: self.update_status("Looking for faces..."))
                        self.main_window.after(0, lambda: self.update_result("No face detected"))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                frame_tk = ImageTk.PhotoImage(frame_pil.resize((640, 480)))

                self.main_window.after(0, lambda: self.update_camera_display(frame_tk))

                time.sleep(0.033)
            except Exception as e:
                print(f"Camera update error: {e}")
                continue

    def process_faces(self, frame, gray, faces):
        if self.paused:
            return
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_roi = gray[y:y + h, x:x + w]

            if len(self.db.face_templates) > 0:
                try:
                    face_resized = cv2.resize(face_roi, (100, 100))
                    face_float = np.float32(face_resized)

                    best_match = None
                    best_score = -1.0

                    for name, template in self.db.face_templates.items():
                        result = cv2.matchTemplate(face_float, template, cv2.TM_CCOEFF_NORMED)
                        max_val = float(np.max(result))
                        print(f"Comparing against {name}: correlation = {max_val:.3f}")
                        if max_val > self.recognition_threshold and max_val > best_score:
                            best_score = max_val
                            best_match = name

                    print(f"Best match: {best_match} with score {best_score:.3f} (threshold: {self.recognition_threshold:.3f})")

                    if best_match is not None:
                        self.current_detection = best_match
                        if not self.paused:
                            self.main_window.after(0, lambda: self.update_status("Face detected!"))
                            self.main_window.after(0, lambda n=best_match: self.update_result(f"Hello, {n}!", True))
                        cv2.putText(
                            frame,
                            f"{best_match} ({best_score:.2f})",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2,
                        )
                    else:
                        self.handle_unknown_face()
                except Exception as e:
                    print(f"Recognition error: {e}")
                    self.handle_unknown_face()
            else:
                if not self.paused:
                    self.main_window.after(0, lambda: self.update_status("Face detected - Database empty"))
                    self.main_window.after(0, lambda: self.update_result("Unknown person"))

    def handle_unknown_face(self):
        if self.paused:
            return
        self.current_detection = None
        self.main_window.after(0, lambda: self.update_status("Face detected"))
        self.main_window.after(0, lambda: self.update_result("Unknown person"))

    def update_camera_display(self, frame_tk):
        self.ui.update_camera_display(frame_tk)

    def update_status(self, text, force=False):
        if self.paused and not force:
            return
        self.ui.update_status(text)

    def update_result(self, text, show_buttons=False, force=False):
        if self.paused and not force:
            return
        self.ui.update_result(text, show_buttons, has_detection=self.current_detection is not None)

    def confirm_identity(self):
        if self.current_detection:
            messagebox.showinfo("Confirmed", f"Welcome, {self.current_detection}!")
            rec = self.attendance.log(self.current_detection)
            print(f"Attendance logged: {rec['name']} at {rec['timestamp']}")
            self.current_detection = None
            self.ui.hide_confirm_buttons()

    def reject_identity(self):
        self.current_detection = None
        self.ui.hide_confirm_buttons()
        self.update_status("Not you. Trying again...", force=True)
        self.update_result("Please try again or add yourself to database", force=True)
        self.paused = True
        self.main_window.after(2000, self.resume_recognition)

    def resume_recognition(self):
        self.update_status("Looking for faces...", force=True)
        self.update_result("No face detected", force=True)
        self.paused = False

    def export_attendance_csv(self):
        if not self.attendance.records:
            messagebox.showinfo("Export Attendance", "No attendance recorded yet.")
            return
        save_path = filedialog.asksaveasfilename(
            title="Save Attendance CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="attendance_export.csv",
        )
        if not save_path:
            return
        try:
            self.attendance.export(save_path)
            messagebox.showinfo("Export Attendance", f"Attendance exported to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Export Attendance", f"Failed to export CSV:\n{e}")

    def add_new_person(self):
        name = simpledialog.askstring("Add Person", "Enter person's name:")
        if not name:
            return
        if name in self.db.known_faces:
            messagebox.showwarning("Warning", "Person already exists in database!")
            return
        self.capture_face_samples(name)

    def capture_face_samples(self, name: str):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "Camera not available!")
            return

        samples_captured = 0
        target_samples = 20
        last_capture_time = 0.0
        capture_delay = 0.5

        messagebox.showinfo(
            "Instructions",
            f"Please look at the camera.\nWe'll capture {target_samples} samples.\n"
            f"Move your head slightly for better recognition.\nPress 'q' to stop early.\n"
            f"Samples will be captured every {capture_delay} seconds."
        )

        face_samples = []

        while samples_captured < target_samples:
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            current_time = time.time()

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face_roi = gray[y:y + h, x:x + w]

                if (w > 50 and h > 50 and current_time - last_capture_time >= capture_delay):
                    face_samples.append(cv2.resize(face_roi, (100, 100)))
                    samples_captured += 1
                    last_capture_time = current_time
                    cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 255), 3)

                cv2.putText(frame, f"Samples: {samples_captured}/{target_samples}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                time_until_next = capture_delay - (current_time - last_capture_time)
                if time_until_next > 0 and samples_captured < target_samples:
                    cv2.putText(frame, f"Next capture in: {time_until_next:.1f}s",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Capturing Face Samples", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        if len(face_samples) < 5:
            messagebox.showwarning("Warning", "Not enough samples captured. Please try again.")
            return

        self.db.save_face_samples(name, face_samples)
        messagebox.showinfo("Success", f"Successfully added {name} to database with {len(face_samples)} samples!")

    def retrain_model(self):
        if not self.db.known_faces:
            messagebox.showwarning("Warning", "No people in database to train on!")
            return
        try:
            self.update_status("Retraining model...")
            start_time = time.time()
            self.db.train_model()
            training_time = time.time() - start_time
            self.update_status("Model retrained successfully!")
            messagebox.showinfo(
                "Success",
                f"Model retrained with {len(self.db.face_templates)} people!\n"
                f"Training completed in {training_time:.2f} seconds."
            )
            self.main_window.after(3000, lambda: self.update_status("Looking for faces..."))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to retrain model: {str(e)}")
            self.update_status("Looking for faces...")

    def view_database(self):
        if not self.db.known_faces:
            messagebox.showinfo("Database", "Database is empty!")
            return
        names = list(self.db.known_faces.keys())
        messagebox.showinfo("Database Contents", "People in database:\n" + "\n".join(names))

    def clear_database(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the entire database?"):
            try:
                self.db.clear()
                messagebox.showinfo("Success", "Database cleared successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear database: {str(e)}")

    def update_threshold(self, value):
        # value comes from Tkinter Scale as a string
        try:
            self.recognition_threshold = int(value) / 100.0
        except Exception:
            pass

    def start(self):
        try:
            self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.main_window.mainloop()
        except Exception as e:
            print(f"Application error: {e}")

    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.main_window.destroy()

