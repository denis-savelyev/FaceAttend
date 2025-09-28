import os
import pickle
from typing import Dict, List
import cv2
import numpy as np


class FaceDatabase:

    def __init__(
        self,
        faces_db_path: str = "faces_db",
        templates_path: str = "face_templates.pkl",
        names_path: str = "names.pkl",
    ) -> None:
        self.faces_db_path = faces_db_path
        self.templates_path = templates_path
        self.names_path = names_path

        # In-memory
        self.known_faces: Dict[str, int] = {}
        self.face_templates: Dict[str, np.ndarray] = {}

        os.makedirs(self.faces_db_path, exist_ok=True)

    def load(self) -> None:

        if os.path.exists(self.names_path):
            try:
                with open(self.names_path, 'rb') as f:
                    self.known_faces = pickle.load(f)
            except Exception as e:
                print(f"Error loading names mapping: {e}")
                self.known_faces = {}
        if os.path.exists(self.templates_path):
            try:
                with open(self.templates_path, 'rb') as f:
                    self.face_templates = pickle.load(f)
            except Exception as e:
                print(f"Error loading templates: {e}")
                self.face_templates = {}

    def save_names(self) -> None:
        with open(self.names_path, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def save_templates(self) -> None:
        with open(self.templates_path, 'wb') as f:
            pickle.dump(self.face_templates, f)

    def save_face_samples(self, name: str, face_samples: List[np.ndarray]) -> None:

        person_dir = os.path.join(self.faces_db_path, name)
        os.makedirs(person_dir, exist_ok=True)

        # Save samples as 0.jpg, 1.jpg, ... (overwrite for fresh captures of a new person)
        for i, face in enumerate(face_samples):
            cv2.imwrite(os.path.join(person_dir, f"{i}.jpg"), face)

        # Update mapping if new
        if name not in self.known_faces:
            self.known_faces[name] = len(self.known_faces)

        # Retrain after saving
        self.train_model()

        # Persist names mapping
        self.save_names()

    def train_model(self) -> None:

        print("Starting model training...")
        self.face_templates = {}

        for name, _ in self.known_faces.items():
            person_dir = os.path.join(self.faces_db_path, name)
            if not os.path.exists(person_dir):
                continue

            person_samples: List[np.ndarray] = []
            for filename in sorted(os.listdir(person_dir)):
                if filename.lower().endswith('.jpg'):
                    img_path = os.path.join(person_dir, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, (100, 100))
                        person_samples.append(np.float32(img_resized))

            if person_samples:
                avg_template = np.mean(person_samples, axis=0)
                self.face_templates[name] = avg_template
                print(f"Trained {name} with {len(person_samples)} samples")

        print(f"Training completed. Total people in database: {len(self.face_templates)}")
        self.save_templates()

    def clear(self) -> None:

        try:
            if os.path.exists(self.faces_db_path):
                import shutil
                shutil.rmtree(self.faces_db_path)
            os.makedirs(self.faces_db_path, exist_ok=True)

            if os.path.exists(self.templates_path):
                os.remove(self.templates_path)
            if os.path.exists(self.names_path):
                os.remove(self.names_path)
        finally:
            self.known_faces = {}
            self.face_templates = {}

