import sqlite3
from typing import List, Dict, Any
import datetime

class UserFeedbackDB:
    def __init__(self, db_path="feedback.db"):
        """
        Initializes a connection to the user feedback database.
        :param db_path: Path to the SQLite database file.
        """
        self.connection = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        """Creates the user feedback table if it does not already exist."""
        query = """
        CREATE TABLE IF NOT EXISTS user_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            pathway_id TEXT NOT NULL,
            responses TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.connection.execute(query)
        self.connection.commit()

    def insert_feedback(self, user_id: str, pathway_id: str, responses: List[str]):
        """
        Inserts user feedback into the database.
        :param user_id: The user's unique identifier.
        :param pathway_id: The ID of the pathway.
        :param responses: List of user response strings from a conversation.
        """
        query = """
        INSERT INTO user_feedback (user_id, pathway_id, responses)
        VALUES (?, ?, ?)
        """
        self.connection.execute(query, (user_id, pathway_id, "\n".join(responses)))
        self.connection.commit()

    def get_recent_conversations(self) -> List[Dict[str, Any]]:
        """
        Retrieves recent user feedback for processing.
        :return: List of dictionaries, each representing a feedback entry.
        """
        query = """
        SELECT user_id, pathway_id, responses, timestamp
        FROM user_feedback
        ORDER BY timestamp DESC
        LIMIT 100
        """
        cursor = self.connection.execute(query)
        recent_feedback = [
            {
                "user_id": row[0],
                "pathway_id": row[1],
                "responses": row[2].split("\n"),
                "timestamp": row[3]
            }
            for row in cursor.fetchall()
        ]
        return recent_feedback

    def close(self):
        """Closes the database connection."""
        self.connection.close()
