def is_valid_move(self, from_pos, to_pos, piece):
	from_x, from_y = from_pos
	to_x, to_y = to_pos
	dx = to_x - from_x
	dy = to_y - from_y
	
	# プレイヤーに応じて前方向を決定
	forward = -1 if piece["player"] == "sente" else 1

	
	# 盤外への移動は無効
	if not self.is_inside_board(to_pos):
		return False
	
	if piece["piece"] == "歩":
		return dx == 0 and dy == forward

	elif piece["piece"] == "香":
		return dx == 0 and dy * forward > 0 and self.is_path_clear(from_pos, to_pos)

	elif piece["piece"] == "桂":
		return abs(dx) == 1 and dy == 2 * forward

	elif piece["piece"] == "銀":
		return (abs(dx) <= 1 and dy == forward) or (abs(dx) == 1 and dy == -forward)

	elif piece["piece"] in ["金", "と", "成香", "成桂", "成銀"]:
		return (abs(dx) <= 1 and dy == forward) or (abs(dx) == 1 and dy == 0) or (dx == 0 and dy == -forward)

	elif piece["piece"] in ["王", "玉"]:
		return abs(dx) <= 1 and abs(dy) <= 1

	elif piece["piece"] == "飛":
		return (dx == 0 or dy == 0) and self.is_path_clear(from_pos, to_pos)

	elif piece["piece"] == "角":
		return abs(dx) == abs(dy) and self.is_path_clear(from_pos, to_pos)

	elif piece["piece"] == "龍":  # 成り飛車
		return (abs(dx) <= 1 and abs(dy) <= 1) or ((dx == 0 or dy == 0) and self.is_path_clear(from_pos, to_pos))

	elif piece["piece"] == "馬":  # 成り角
		return (abs(dx) <= 1 and abs(dy) <= 1) or (abs(dx) == abs(dy) and self.is_path_clear(from_pos, to_pos))

	return False  # デフォルトでは無効な移動とする

# 飛車、角行、香車の動きで使用され、移動経路に他の駒がないかをチェック
def is_path_clear(self, from_pos, to_pos):
	from_x, from_y = from_pos
	to_x, to_y = to_pos
	dx = to_x - from_x
	dy = to_y - from_y
	steps = max(abs(dx), abs(dy))

	for i in range(1, steps):
		x = from_x + i * (dx // steps)
		y = from_y + i * (dy // steps)
		if self.board[y][x] is not None:
			return False
	return True

"""
駒が成れるかどうかを判断するメソッド
引数：pos(駒の現在位置)、piece（駒の情報「プレイヤーや種類」)
"""
def can_promote(self, pos, piece):
	x, y = pos
	promotable_pieces = ["歩", "香車", "桂馬", "銀", "飛車", "角"]

	if piece["piece"] not in promotable_pieces:
		return False

	if piece["player"] == "sente":
		return y < 3  # 3段目以内
	else:  # gote
		return y > 5  # 7段目以上

def is_inside_board(self, pos):
	# 指定された位置が盤面内かチェック
	x, y = pos
	return 0 <= x < 9 and 0 <= y < 9

# 指定された駒を成り駒に変換する
def promote_piece(self, piece):
	promotion_map = {
		"歩": "と", "香車": "成香", "桂馬": "成桂", "銀": "成銀",
		"飛車": "龍", "角": "馬"
	}
	return promotion_map.get(piece, piece)

#成り駒を戻す
def unpromote_piece(self, piece):
	unpromotion_map = {
		"と": "歩", "成香": "香車", "成桂": "桂馬", "成銀": "銀",
		"龍": "飛車", "馬": "角"
	}
	return unpromotion_map.get(piece, piece)

# 二歩チェック
# 指定された列を調べ、同じプレイヤーの歩がすでにあ
def is_nifu(self, pos, player):
	x, _ = pos
	for y in range(9):
		if self.board[y][x] and self.board[y][x]["piece"] == "歩" and self.board[y][x]["player"] == player:
			return True
	return False

"""
現在の盤面状態で、指定されたプレイヤーの王が王手状態かチェック
"""
def is_check(self, player):
	# king_pos = self.find_king(player)
	# opponent = self.get_opponent(player)
	# for y in range(9):
	#     for x in range(9):
	#         piece = self.board[y][x]
	#         if piece and piece["player"] == opponent:
	#             if self.is_valid_move((x, y), king_pos, piece):
	#                 return True
	return self.is_check_on_board(player,self.board)

"""
指定されたプレイヤーが詰みかどうかをチェック
"""
def is_checkmate(self, player):
	# プレイヤーが王手状態にあるかをチェック
	if not self.is_check(player):
		return False
	
	# 盤上の駒の移動による王手回避の可能性チェック
	for y in range(9):
		for x in range(9):
			piece = self.board[y][x]
			if piece and piece["player"] == player:
				for to_y in range(9):
					for to_x in range(9):
						# 合法手であれば、
						if self.is_valid_move((x, y), (to_x, to_y), piece):
							# 仮の移動を行う
							temp_board = [row[:] for row in self.board]
							temp_board[to_y][to_x] = piece
							temp_board[y][x] = None
							
							# 王手が回避できるかチェック
							if not self.is_check_on_board(player, temp_board):
								return False
	
	# 持ち駒を使って王手を回避できるかチェック
	for piece in self.captured_pieces[player]:
		for y in range(9):
			for x in range(9):
				# 全ての持ち駒について、盤上のすべての位置に打てるかをチェック
				if self.can_drop_piece(piece, (x, y), player):
					# 打てる場合、持ち駒を打った後の盤面で王手が回避できていれば詰みではない
					temp_board = [row[:] for row in self.board]
					temp_board[y][x] = {"piece": piece, "player": player}
					if not self.is_check_on_board(player, temp_board):
						return False
	
	return True

"""
「現在の盤面上」で指定されたプレイヤーの王の位置を見つける
"""
def find_king(self, player):
	return self.find_king_on_board(player, self.board)

"""
「与えられた盤面上」で指定されたプレイヤーの王の位置を見つける
王手や詰みの判定に用いる
"""
def find_king_on_board(self, player, board):
	for y in range(9):
		for x in range(9):
			piece = board[y][x]
			if piece and piece["piece"] in ["王", "玉"] and piece["player"] == player:
				return (x, y)
	raise ValueError(f"King not found for player {player}")

"""
与えられたプレイヤーの相手（senteならgote）を返す関数
手番の交代や相手プレイヤーに関する処理に用いる
"""
def get_opponent(self, player):
	return "gote" if player == "sente" else "sente"

"""
指定された駒（持ち駒）を特定の位置に打つことができるかチェック
"""
def can_drop_piece(self, piece, pos, player): # Piece:打とうとしている駒、pos:打とうとしている位置、player:打とうとしているプレイヤー
	x, y = pos
	
	# 駒を打つ場所に既に駒がある場合は打てない
	if self.board[y][x] is not None:
		return False
	
	# 歩に関する特別ルール
	if piece == "歩":
		# 二歩の禁止
		if self.is_nifu((x, y), player):
			return False
		
		# 最奥の段には打てない
		if (player == "sente" and y == 0) or (player == "gote" and y == 8):
			return False
		
		# 打ち歩詰めの禁止
		if self.is_pawn_drop_checkmate(pos, player):
			return False
	
	# 香車と桂馬に関する特別ルール
	if piece == "香車":
		if (player == "sente" and y == 0) or (player == "gote" and y == 8):
			return False
	elif piece == "桂馬":
		if (player == "sente" and y <= 1) or (player == "gote" and y >= 7):
			return False
	
	return True

# 打ち歩詰め（禁じ手）のチェック
def is_pawn_drop_checkmate(self, pos, player):
	x, y = pos
	opponent = self.get_opponent(player) # 相手プレイヤーを特定
	
	# 一時的に歩を配置（実際の場面は変更しない）
	self.board[y][x] = {"piece": "歩", "player": player}
	
	# 相手の王が詰んでいるかチェック
	is_checkmate = self.is_checkmate(opponent)
	
	# 配置した歩を元に戻す
	self.board[y][x] = None
	
	return is_checkmate
