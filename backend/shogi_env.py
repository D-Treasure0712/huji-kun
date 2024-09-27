# shogi_environment.py

import numpy as np
from cshogi import Board, Move, PIECE_EMPTY, BLACK, WHITE, CSA, move_to_usi
import random

class ShogiEnv:
    def __init__(self):
        self.board = Board()
        self.done = False
        self.winner = None
        self.observation_space_shape = (104, 9, 9)  # 特徴量の次元
        self.action_space_size = 27 * 9 * 9 * 9 * 9 + 7 * 9 * 9  # 最大の行動数
        self.init_move_list()
        self.reset()
    
    def reset(self):
        self.board.reset()
        self.done = False
        self.winner = None
        return self.get_observation()
    
    def get_observation(self):
        """
        盤面の状態を特徴量として取得します。
        特徴量は以下の通りです。
        - 駒の配置（14種類 × 9 × 9）
        - 手番（1種類 × 9 × 9）
        - 持ち駒（18種類 × 9 × 9）
        合計で33チャンネルの9×9の特徴量となります。
        """
        # 駒の配置
        piece_planes = np.zeros((14, 9, 9), dtype=np.float32)
        for sq in range(81):
            piece = self.board.piece_at(sq)
            if piece != PIECE_EMPTY:
                color = self.board.color_at(sq)
                piece_type = piece & 0b1111
                plane_idx = (piece_type - 1) + (7 if color == WHITE else 0)
                piece_planes[plane_idx, sq // 9, sq % 9] = 1.0
        # 手番の情報
        turn_plane = np.full((1, 9, 9), self.board.turn, dtype=np.float32)
        # 持ち駒の情報
        hands_planes = np.zeros((18, 9, 9), dtype=np.float32)
        # 先手の持ち駒
        for piece_type in range(1, 8):
            count = self.board.hand_n_black(piece_type)
            if count > 0:
                plane_idx = piece_type - 1
                hands_planes[plane_idx, :, :] = count / 18.0  # 正規化
        # 後手の持ち駒
        for piece_type in range(1, 8):
            count = self.board.hand_n_white(piece_type)
            if count > 0:
                plane_idx = piece_type + 7
                hands_planes[plane_idx, :, :] = count / 18.0  # 正規化
        # 特徴量を結合
        observation = np.concatenate([piece_planes, turn_plane, hands_planes], axis=0)
        return observation  # shape: (33, 9, 9)
    
    def step(self, move):
        """
        指し手を適用します。
        """
        if move not in self.board.legal_moves:
            raise ValueError("Illegal move attempted.")
        self.board.push(move)
        reward = 0
        self.done = False
        if self.board.is_game_over():
            self.done = True
            if self.board.is_draw():
                reward = 0
            else:
                winner = 'WHITE' if self.board.turn == BLACK else 'BLACK'
                self.winner = winner
                reward = 1 if winner == 'BLACK' else -1
        return self.get_observation(), reward, self.done, {}
    
    def legal_moves(self):
        """
        現在の局面での合法手のリストを返します。
        """
        return list(self.board.legal_moves)
    
    def render(self):
        """
        現在の盤面を表示します。
        """
        print(self.board.kif_str())
    
    def init_move_list(self):
        """
        全ての可能な指し手のリストを作成します。
        9×9の盤上で、駒の種類、移動元、移動先、成りなどの組み合わせを考慮します。
        """
        self.move_list = []
        self.move_to_action = {}
        self.action_to_move = {}
        action_index = 0

        # 全ての移動元、移動先の組み合わせを考慮
        for from_sq in range(81):
            for to_sq in range(81):
                # 成り・不成りの両方を考慮
                for promote in [False, True]:
                    move = Move(from_sq, to_sq, promote)
                    # 初期局面での合法手でなくても、全ての可能な手をリストに含めます
                    if Move.is_drop(move):
                        continue  # 打つ手は後で追加
                    move_csa = CSA.move_to_csa(move)
                    self.move_list.append(move)
                    self.move_to_action[move_csa] = action_index
                    self.action_to_move[action_index] = move
                    action_index += 1
        # 持ち駒を打つ手
        for to_sq in range(81):
            for piece_type in range(1, 8):
                move = Move(0, to_sq, False, piece_type=piece_type)
                move_csa = CSA.move_to_csa(move)
                self.move_list.append(move)
                self.move_to_action[move_csa] = action_index
                self.action_to_move[action_index] = move
                action_index += 1
        
        self.action_space_size = action_index
    
    def encode_action(self, move):
        """
        指し手を行動にエンコードします。
        """
        move_csa = CSA.move_to_csa(move)
        action = self.move_to_action.get(move_csa)
        return action
    
    def decode_action(self, action):
        """
        行動を指し手にデコードします。
        """
        move = self.action_to_move.get(action)
        return move
