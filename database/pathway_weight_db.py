import sqlite3

class PathwayWeightDB:
    def __init__(self, db_path="pathway_weights.db"):
        """
        Initializes a connection to the pathway weights database.
        :param db_path: Path to the SQLite database file.
        """
        self.connection = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        """Creates the pathway weights table if it does not already exist."""
        query = """
        CREATE TABLE IF NOT EXISTS pathway_weights (
            pathway_id TEXT PRIMARY KEY,
            weight REAL NOT NULL
        )
        """
        self.connection.execute(query)
        self.connection.commit()

    def get_weight(self, pathway_id: str) -> float:
        """
        Retrieves the weight for a specific pathway.
        :param pathway_id: The ID of the pathway.
        :return: Weight of the pathway.
        """
        query = "SELECT weight FROM pathway_weights WHERE pathway_id = ?"
        cursor = self.connection.execute(query, (pathway_id,))
        result = cursor.fetchone()
        return result[0] if result else 0.5  # Default weight is 0.5 if not found

    def update_weight(self, pathway_id: str, new_weight: float):
        """
        Updates the weight for a specific pathway.
        :param pathway_id: The ID of the pathway.
        :param new_weight: New weight to set for the pathway.
        """
        query = """
        INSERT INTO pathway_weights (pathway_id, weight)
        VALUES (?, ?)
        ON CONFLICT(pathway_id) DO UPDATE SET weight = excluded.weight
        """
        self.connection.execute(query, (pathway_id, new_weight))
        self.connection.commit()

    def close(self):
        """Closes the database connection."""
        self.connection.close()
