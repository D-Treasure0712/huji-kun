from flask import Flask, request, jsonify
from cshogi import *

app = Flask(__name__)

@app.route('/api/chack_checkmate', methods = ['POST'])

def check_checkmate():
    board = Board()
    
    # リクエストからのSFEN形式の盤面表現を取得
    sfen = request.json['sfen']
    
    # cshogiのBoardオブジェクトを作成し、SFENで盤面をセット
    board.set_sfen(sfen)
    
    legal_moves = list(board.legal_moves)
    
    # 王手かどうかをチェック
    is_check = board.is_check()
    
    # 詰みかどうかをチェック
    is_checkmate = board.is_game_over()
    
    return jsonify({
        'is_checkmate': is_checkmate,
        'is_check' : is_check,
#        'legal_moves': [cshogi.move_to_usi(move) for move in board.legal_moves],
#        'is_game_over': board.is_game_over(),
#        'turn': 'sente' if board.turn == cshogi.BLACK else 'gote'
    })


if __name__ == '__main__':
    app.run(debug=True)