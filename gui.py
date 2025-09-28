import tkinter as tk
from typing import Callable, Optional


class AppUI:
    # all tkinter widgets and ui updates

    def __init__(
        self,
        root: tk.Tk,
        *,
        on_confirm: Callable[[], None],
        on_reject: Callable[[], None],
        on_add_person: Callable[[], None],
        on_retrain: Callable[[], None],
        on_view_db: Callable[[], None],
        on_clear_db: Callable[[], None],
        on_export_attendance: Callable[[], None],
        on_threshold_change: Callable[[str], None],
        initial_threshold: float,
    ) -> None:
        self.root = root
        self.on_confirm = on_confirm
        self.on_reject = on_reject
        self.on_add_person = on_add_person
        self.on_retrain = on_retrain
        self.on_view_db = on_view_db
        self.on_clear_db = on_clear_db
        self.on_export_attendance = on_export_attendance
        self.on_threshold_change = on_threshold_change

        # Widget refs
        self.camera_label: Optional[tk.Label] = None
        self.status_label: Optional[tk.Label] = None
        self.result_label: Optional[tk.Label] = None
        self.confirm_button: Optional[tk.Button] = None
        self.reject_button: Optional[tk.Button] = None

        self._build(initial_threshold)

    def update_camera_display(self, frame_tk) -> None:
        if self.camera_label is not None:
            self.camera_label.configure(image=frame_tk)
            self.camera_label.image = frame_tk

    def update_status(self, text: str) -> None:
        if self.status_label is not None:
            self.status_label.configure(text=text)

    def update_result(self, text: str, show_buttons: bool, has_detection: bool) -> None:
        if self.result_label is not None:
            self.result_label.configure(text=text)
        if self.confirm_button and self.reject_button:
            if show_buttons and has_detection:
                self.confirm_button.pack(pady=5)
                self.reject_button.pack(pady=5)
            else:
                self.confirm_button.pack_forget()
                self.reject_button.pack_forget()

    def hide_confirm_buttons(self) -> None:
        if self.confirm_button and self.reject_button:
            self.confirm_button.pack_forget()
            self.reject_button.pack_forget()

    def _build(self, initial_threshold: float) -> None:
        self.root.geometry("1200x900")
        self.root.title("FaceAttend - Facial Recognition System")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)

        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        title_label = tk.Label(
            main_frame,
            text="FaceAttend - Facial Recognition System",
            font=("Arial", 20, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
        )
        title_label.pack(pady=(0, 20))

        content_frame = tk.Frame(main_frame, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True)

        camera_frame = tk.Frame(content_frame, bg="#34495e", relief=tk.RAISED, bd=2)
        camera_frame.pack(side=tk.LEFT, padx=(0, 20))

        self.camera_label = tk.Label(camera_frame, bg="#34495e", width=640, height=480)
        self.camera_label.pack(padx=10, pady=10)

        control_frame = tk.Frame(content_frame, bg="#f0f0f0", width=350)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        control_frame.pack_propagate(False)

        status_frame = tk.LabelFrame(
            control_frame,
            text="Status",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=10,
            pady=10,
        )
        status_frame.pack(fill=tk.X, pady=(0, 20))

        self.status_label = tk.Label(
            status_frame,
            text="Looking for faces...",
            font=("Arial", 14),
            bg="#f0f0f0",
            fg="#7f8c8d",
        )
        self.status_label.pack()

        result_frame = tk.LabelFrame(
            control_frame,
            text="Recognition Result",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=10,
            pady=10,
        )
        result_frame.pack(fill=tk.X, pady=(0, 20))

        self.result_label = tk.Label(
            result_frame,
            text="No face detected",
            font=("Arial", 16),
            bg="#f0f0f0",
            fg="#34495e",
            wraplength=300,
        )
        self.result_label.pack(pady=10)

        confirm_frame = tk.Frame(result_frame, bg="#f0f0f0", height=80)
        confirm_frame.pack(pady=10)
        confirm_frame.pack_propagate(True)

        self.confirm_button = tk.Button(
            confirm_frame,
            text="Yes, that's me!",
            font=("Arial", 11, "bold"),
            bg="#27ae60",
            fg="white",
            command=self.on_confirm,
            width=18,
            height=1,
        )
        self.reject_button = tk.Button(
            confirm_frame,
            text="Not me",
            font=("Arial", 11),
            bg="#e74c3c",
            fg="white",
            command=self.on_reject,
            width=18,
            height=2,
        )
        # Initially hidden; update_result will manage visibility
        self.confirm_button.pack_forget()
        self.reject_button.pack_forget()

        db_frame = tk.LabelFrame(
            control_frame,
            text="Database Management",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=10,
            pady=10,
        )
        db_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Button(
            db_frame,
            text="Add New Person",
            font=("Arial", 10),
            bg="#3498db",
            fg="white",
            command=self.on_add_person,
            padx=15,
            pady=5,
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            db_frame,
            text="Retrain Model",
            font=("Arial", 10),
            bg="#f39c12",
            fg="white",
            command=self.on_retrain,
            padx=15,
            pady=5,
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            db_frame,
            text="View Database",
            font=("Arial", 10),
            bg="#9b59b6",
            fg="white",
            command=self.on_view_db,
            padx=15,
            pady=5,
        ).pack(fill=tk.X, pady=2)

        tk.Button(
            db_frame,
            text="Clear Database",
            font=("Arial", 10),
            bg="#e67e22",
            fg="white",
            command=self.on_clear_db,
            padx=15,
            pady=5,
        ).pack(fill=tk.X, pady=2)

        attendance_frame = tk.LabelFrame(
            control_frame,
            text="Attendance",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=10,
            pady=10,
        )
        attendance_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Button(
            attendance_frame,
            text="Export Attendance CSV",
            font=("Arial", 10),
            bg="#2ecc71",
            fg="white",
            command=self.on_export_attendance,
            padx=15,
            pady=5,
        ).pack(fill=tk.X, pady=2)

        settings_frame = tk.LabelFrame(
            control_frame,
            text="Settings",
            font=("Arial", 12, "bold"),
            bg="#f0f0f0",
            fg="#2c3e50",
            padx=10,
            pady=10,
        )
        settings_frame.pack(fill=tk.X)

        tk.Label(settings_frame, text="Recognition Sensitivity:", font=("Arial", 10), bg="#f0f0f0").pack()

        threshold_var = tk.IntVar(value=int(initial_threshold * 100))
        tk.Scale(
            settings_frame,
            from_=20,
            to=100,
            variable=threshold_var,
            orient=tk.HORIZONTAL,
            command=self.on_threshold_change,
            bg="#f0f0f0",
        ).pack(fill=tk.X, pady=5)

        tk.Label(settings_frame, text="Lower = More Strict", font=("Arial", 8), bg="#f0f0f0", fg="#7f8c8d").pack()

