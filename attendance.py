import os
import csv
import time
from typing import List, Dict, Optional


class AttendanceManager:
    # Manage attendance records

    def __init__(self, log_path: str = "attendance_log.csv") -> None:
        self.log_path = log_path
        self.records: List[Dict[str, str]] = []
        self.load_records()

    def load_records(self) -> None:
        # Load attendance records from CSV if it exists
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, mode='r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "name" in row and "timestamp" in row:
                        self.records.append({
                            "name": row["name"],
                            "timestamp": row["timestamp"],
                        })
        except Exception as e:
            print(f"Failed to load attendance CSV: {e}")

    def log(self, name: str, timestamp: Optional[str] = None) -> Dict[str, str]:
        # Append an attendance record, Returns the record
        if timestamp is None:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        record = {"name": name, "timestamp": timestamp}
        self.records.append(record)

        # Append to CSV (create with header if missing)
        file_exists = os.path.exists(self.log_path)
        try:
            with open(self.log_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["name", "timestamp"])
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
        except Exception as e:
            print(f"Failed to write attendance CSV: {e}")

        return record

    def export(self, save_path: str) -> None:
        # Export current in memory records to a given CSV path
        with open(save_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["name", "timestamp"])
            writer.writeheader()
            writer.writerows(self.records)

