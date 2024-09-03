from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Game(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    board_state = db.Column(db.String, nullable=False)
    current_player = db.Column(db.String, nullable=False)
    move_history = db.Column(db.String, nullable=False)
    status = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __init__(self, board_state, current_player="sente", move_history="", status="ongoing"):
        self.board_state = board_state
        self.current_player = current_player
        self.move_history = move_history
        self.status = status

    def to_dict(self):
        return {
            'id': self.id,
            'board_state': self.board_state,
            'current_player': self.current_player,
            'move_history': self.move_history,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }