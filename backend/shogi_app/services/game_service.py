from models.shogi_board import ShogiBoard
from utils.shogi_rules import is_valid_move, is_check, is_checkmate, can_promote, can_drop_piece

class GameService:
    def __init__(self):
        self.board = ShogiBoard()
        self.current_player = "sente"

    def make_move(self, from_pos, to_pos):
        if not is_valid_move(self.board, from_pos, to_pos, self.current_player):
            raise ValueError("Invalid move")

        self.board.move_piece(from_pos, to_pos)
        
        if can_promote(self.board.get_piece(to_pos), from_pos, to_pos, self.current_player):
            # プロモーションの処理（ユーザーの選択を受け付ける必要がある）
            pass

        if is_checkmate(self.board, self.get_opponent()):
            return "checkmate"
        elif is_check(self.board, self.get_opponent()):
            return "check"

        self.switch_player()
        return "continue"

    def drop_piece(self, piece, to_pos):
        if not can_drop_piece(self.board, to_pos, piece, self.current_player):
            raise ValueError("Invalid drop")

        # 持ち駒から駒を取り除き、盤面に配置
        self.board.captured_pieces[self.current_player].remove(piece)
        self.board.board[to_pos[1]][to_pos[0]] = {"piece": piece, "player": self.current_player}

        self.switch_player()

    def switch_player(self):
        self.current_player = self.get_opponent()

    def get_opponent(self):
        return "gote" if self.current_player == "sente" else "sente"