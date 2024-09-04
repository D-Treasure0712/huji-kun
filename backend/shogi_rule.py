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
    
    # 王手かどうかをチェック
    is_check = board.is_check()
    
    if (board.turn == 0):
        board.turn = 1
    elif (board.turn == 1):
        board.turn = 0

    is_check2 = board.is_check()
    print(is_check)
    print(is_check2)
    
    # 詰みかどうかをチェック
    is_checkmate = board.is_game_over()
    
    return jsonify({
        'is_checkmate': is_checkmate,
        'is_check' : is_check,
        'is_check2' : is_check2,
    })


if __name__ == '__main__':
    app.run(debug=True)