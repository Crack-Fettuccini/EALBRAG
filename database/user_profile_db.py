import sqlite3
from typing import Optional

class UserProfileDB:
    def __init__(self, connection_string: str):
        """
        Initialize the UserProfileDB with a connection string.
        :param connection_string: Database connection string.
        """
        self.connection_string = connection_string
        self.conn = sqlite3.connect(self.connection_string, check_same_thread=False)
        self._create_table()
    
    def _create_table(self):
        """
        Create the user_profiles table if it doesn't exist.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ground_truth TEXT NOT NULL
        );
        """
        cursor = self.conn.cursor()
        cursor.execute(create_table_query)
        self.conn.commit()
    
    def add_user_profile(self, ground_truth: str):
        """
        Add a new user profile.
        :param ground_truth: The generated ground truth text.
        """
        insert_query = "INSERT INTO user_profiles (ground_truth) VALUES (?);"
        cursor = self.conn.cursor()
        cursor.execute(insert_query, (ground_truth,))
        self.conn.commit()
    
    def update_user_profile(self, ground_truth: str):
        """
        Update the user profile with new ground truth information.
        For simplicity, this example appends the new ground truth.
        :param ground_truth: The generated ground truth text.
        """
        # Fetch existing profiles
        select_query = "SELECT ground_truth FROM user_profiles ORDER BY user_id DESC LIMIT 1;"
        cursor = self.conn.cursor()
        cursor.execute(select_query)
        result = cursor.fetchone()
        if result:
            updated_gt = result[0] + "\n" + ground_truth
            update_query = "UPDATE user_profiles SET ground_truth = ? WHERE user_id = ?;"
            cursor.execute(update_query, (updated_gt, cursor.lastrowid))
        else:
            self.add_user_profile(ground_truth)
        self.conn.commit()
    
    def get_user_profile(self, user_id: int) -> Optional[str]:
        """
        Retrieve a user profile by user_id.
        :param user_id: The ID of the user.
        :return: Ground truth text or None if not found.
        """
        select_query = "SELECT ground_truth FROM user_profiles WHERE user_id = ?;"
        cursor = self.conn.cursor()
        cursor.execute(select_query, (user_id,))
        result = cursor.fetchone()
        return result[0] if result else None
