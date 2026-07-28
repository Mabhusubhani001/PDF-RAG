import os
import uuid
from typing import Optional


class TempStorageManager:
    """
    Manages temporary PDF document uploads on disk.
    """
    def __init__(self, base_dir: str = "temp_uploads"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_temp_file(self, file_bytes: bytes, original_filename: str) -> str:
        """
        Saves uploaded file bytes to temporary workspace directory.
        """
        unique_name = f"{uuid.uuid4().hex}_{original_filename}"
        file_path = os.path.join(self.base_dir, unique_name)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
        return file_path

    def delete_temp_file(self, file_path: str) -> bool:
        """
        Removes temporary PDF file after embedding generation.
        """
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
