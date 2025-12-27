"""
SQLite storage pro ukládání agregovaných metrik.
"""
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class MetricsStorage:
    """
    SQLite databáze pro ukládání agregovaných metrik.
    
    Ukládá pouze agregace po minutách, ne raw data.
    """
    
    def __init__(self, db_path: Path):
        """
        Args:
            db_path: Cesta k SQLite databázi
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Inicializuje databázi a vytvoří tabulky pokud neexistují."""
        db_path = self.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Tabulka pro agregované metriky
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics_minute (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                occupancy_avg REAL NOT NULL,
                occupancy_max INTEGER NOT NULL,
                crossings INTEGER NOT NULL DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Index pro rychlé dotazy
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON metrics_minute(timestamp)
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized: {db_path}")
    
    def insert_minute_aggregate(
        self,
        timestamp: datetime,
        occupancy_avg: float,
        occupancy_max: int,
        crossings: int
    ):
        """
        Uloží agregaci za 1 minutu.
        
        Args:
            timestamp: Časové razítko (začátek minuty)
            occupancy_avg: Průměrná occupancy
            occupancy_max: Maximální occupancy
            crossings: Počet crossingů za minutu
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO metrics_minute (timestamp, occupancy_avg, occupancy_max, crossings)
            VALUES (?, ?, ?, ?)
        """, (timestamp, occupancy_avg, occupancy_max, crossings))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Saved minute aggregate: {timestamp}, avg={occupancy_avg:.1f}, max={occupancy_max}, crossings={crossings}")
    
    def get_timeseries(self, minutes: int = 60) -> List[Dict]:
        """
        Získá časovou řadu metrik za posledních N minut.
        
        Args:
            minutes: Počet minut zpět
            
        Returns:
            Seznam slovníků s metrikami
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, occupancy_avg, occupancy_max, crossings
            FROM metrics_minute
            WHERE timestamp >= datetime('now', '-{} minutes')
            ORDER BY timestamp ASC
        """.format(minutes))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = []
        for row in rows:
            result.append({
                'timestamp': row[0],
                'occupancy_avg': row[1],
                'occupancy_max': row[2],
                'crossings': row[3]
            })
        
        return result
    
    def get_latest(self) -> Dict:
        """
        Získá nejnovější záznam.
        
        Returns:
            Slovník s metrikami nebo prázdný slovník
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT timestamp, occupancy_avg, occupancy_max, crossings
            FROM metrics_minute
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'timestamp': row[0],
                'occupancy_avg': row[1],
                'occupancy_max': row[2],
                'crossings': row[3]
            }
        else:
            return {}
    
    def cleanup_old_data(self, days: int = 7):
        """
        Smaže stará data starší než N dní.
        
        Args:
            days: Počet dní
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM metrics_minute
            WHERE timestamp < datetime('now', '-{} days')
        """.format(days))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old records")
