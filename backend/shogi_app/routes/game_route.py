from flask import Blueprint, request, jsonify
from services.game_service import GameService

game_routes = Blueprint('game_routes', __name__)
game_service = GameService()

@game_routes.route('/move', methods=['POST'])
def make_move():
    data = request.json
    from_pos = tuple(data['from'])
    to_pos = tuple(data['to'])

    try:
        result = game_service.make_move(from_pos, to_pos)
        return jsonify({
            "result": result,
            "board": game_service.board.board,
            "captured_pieces": game_service.board.captured_pieces,
            "current_player": game_service.current_player
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@game_routes.route('/drop', methods=['POST'])
def drop_piece():
    data = request.json
    piece = data['piece']
    to_pos = tuple(data['to'])

    try:
        game_service.drop_piece(piece, to_pos)
        return jsonify({
            "board": game_service.board.board,
            "captured_pieces": game_service.board.captured_pieces,
            "current_player": game_service.current_player
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400