import { Piece, Blank, King, Rook, Bishop, GoldGeneral, SilverGeneral, Knight, Lance, Pawn,  PromotedPawn, PromotedLance, PromotedKnight, PromotedSilverGeneral, PromotedBishop, PromotedRook } from './Pieces';

class BoardInfo {

    constructor() {
        this.turn = "先手";
        this.board = [[new Lance("後手"), new Knight("後手"), new SilverGeneral("後手"), new GoldGeneral("後手"), new King("後手"), new GoldGeneral("後手"), new SilverGeneral("後手"), new Knight("後手"), new Lance("後手")],
        [new Blank(), new Rook("後手"), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Bishop("後手"), new Blank()],
        [new Pawn("後手"), new Pawn("後手"), new Pawn("後手"), new Pawn("後手"), new Pawn("後手"), new Pawn("後手"), new Pawn("後手"), new Pawn("後手"), new Pawn("後手")],
        [new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank()],
        [new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank()],
        [new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank()],
        [new Pawn("先手"), new Pawn("先手"), new Pawn("先手"), new Pawn("先手"), new Pawn("先手"), new Pawn("先手"), new Pawn("先手"), new Pawn("先手"), new Pawn("先手")],
        [new Blank(), new Bishop("先手"), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Rook("先手"), new Blank()],
        [new Lance("先手"), new Knight("先手"), new SilverGeneral("先手"), new GoldGeneral("先手"), new King("先手"), new GoldGeneral("先手"), new SilverGeneral("先手"), new Knight("先手"), new Lance("先手")]
        ];
        this.selection = new Selection();
        this.pieceStandNum = {
            "先手": { "歩": 0, "香": 0, "桂": 0, "銀": 0, "金": 0, "角": 0, "飛": 0 },
            "後手": { "歩": 0, "香": 0, "桂": 0, "銀": 0, "金": 0, "角": 0, "飛": 0 }
        };
        this.pieceStand = {
            "先手": [new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank()],
            "後手": [new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank(), new Blank()]
        };
        this.is_checkmate = false;
        this.isGameOver = false;
        this.winner = null;
    }

    async checkCheckmate() {
        // 現在の盤面状態をSFEN形式に変換
        const sfen = this.convertToSFEN();
        
        // デバッグ用：APIに送信するSFENをコンソールに出力
        console.log('Sending SFEN to API:', sfen);
     
        try {
            // '/api/chack_checkmate'エンドポイントにPOSTリクエストを送信
            // 注意: 'check'のスペルミスがありますが、APIの実装に合わせています
            const response = await fetch('/api/chack_checkmate', {
                method: 'POST',  // HTTPメソッドをPOSTに設定
                headers: {
                    'Content-Type': 'application/json',  // コンテンツタイプをJSONに指定
                },
                body: JSON.stringify({ sfen: sfen }),  // SFENをJSON形式でリクエストボディに設定
            });
     
            // レスポンスをJSON形式で解析
            const data = await response.json();
     
            // 王手状態をis_checkプロパティに設定
            this.is_check = data.is_check;
            this.is_check2 = data.is_check2;
     
            // 詰み（チェックメイト）の場合の処理
            if (data.is_checkmate) {
                // ゲーム終了フラグを立てる
                this.isGameOver = true;
                // 勝者を設定（現在の手番の逆）
                this.winner = this.turn === "先手" ? "後手" : "先手";
            }
        } catch (error) {
            // エラーが発生した場合、コンソールにエラーメッセージを出力
            console.error('Error checking for checkmate:', error);
        }
     }

    convertToSFEN() {
        let sfen = '';
        
        // 1. 盤面の状態
        for (let i = 0; i < 9; i++) {
            let emptyCount = 0;
            for (let j = 0; j < 9; j++) {
                const piece = this.board[i][j];
                if (piece instanceof Blank) {
                    emptyCount++;
                } else {
                    if (emptyCount > 0) {
                        sfen += emptyCount;
                        emptyCount = 0;
                    }
                    sfen += this.pieceToSFEN(piece);
                }
            }
            if (emptyCount > 0) {
                sfen += emptyCount;
            }
            if (i < 8) sfen += '/';
        }

        // 2. 手番
        sfen += ` ${this.turn === "先手" ? 'b' : 'w'}`;

        // 3. 持ち駒
        let capturedPieces = '';
        for (const player of ["先手", "後手"]) {
            for (const [pieceName, count] of Object.entries(this.pieceStandNum[player])) {
                if (count > 0) {
                    if (count > 1) {
                        capturedPieces += count;
                    }
                    capturedPieces += this.pieceToSFEN(Piece.getPieceByName(pieceName, player));
                }
            }
        }
        sfen += capturedPieces ? ` ${capturedPieces}` : ' -';

        // 4. 手数 (このクラスで手数を追跡していない場合、1とします)
        sfen += ' 1';

        return sfen;
    }


    pieceToSFEN(piece) {
        const pieceSymbols = {
            "玉": "K", "飛": "R", "角": "B", "金": "G", "銀": "S", "桂": "N", "香": "L", "歩": "P",
            "竜": "+R", "馬": "+B", "成銀": "+S", "成桂": "+N", "成香": "+L", "と": "+P"
        };
        let symbol = pieceSymbols[piece.name];
        return piece.owner === "後手" ? symbol.toLowerCase() : symbol;
    }
    async boardClick(i, j) {
        // 駒を動かす前の状態を保存
        const previousState = this.saveCurrentState();

        // 駒が既に選択されている状態の場合
        if (this.selection.state) {
            // クリックされたマスが「配置可能」でない場合は何もせずに終了
            if (this.selection.boardSelectInfo[i][j] !== "配置可能") {
                return;
            }


            let myPiece;
     
            // 持ち駒を使用する場合
            if (this.selection.pieceStandPiece.name) {
                // 選択された持ち駒を取得
                myPiece = this.selection.pieceStandPiece;
                // 持ち駒の数を1減らす
                this.pieceStandNum[this.turn][myPiece.name] -= 1;
                // 持ち駒の表示を更新
                this.makePieceStand();
            } 
            // 盤上の駒を動かす場合
            else {
                // 選択された駒を取得
                myPiece = this.board[this.selection.before_i][this.selection.before_j];
                // 元の位置を空にする
                this.board[this.selection.before_i][this.selection.before_j] = new Blank();
                
                // 移動先の駒（相手の駒）を取得
                let yourPiece = this.board[i][j];
                
                // 相手の駒がある場合
                if (yourPiece.name) {
                    // 成り駒の場合、元の駒に戻す
                    if (yourPiece.getPiece()) {
                        yourPiece = yourPiece.getPiece();
                    }
                    // 取った駒を持ち駒に追加
                    this.pieceStandNum[myPiece.owner][yourPiece.name] += 1;
                    // 持ち駒の表示を更新
                    this.makePieceStand();
                }
     
                // 駒の移動が可能な場合、成りの判定を行う
                if (this.existCanMove(i, j, myPiece)) {
                    myPiece = this.checkPromote(myPiece, i, this.selection.before_i);
                } 
                // 移動不可能な場合（飛車や角が端に到達した場合など）は強制的に成る
                else {
                    myPiece = myPiece.getPromotedPiece();
                }
            }
     
            // 新しい位置に駒を配置
            this.board[i][j] = myPiece;
            // 手番を交代
            this.turn = this.turn === "先手" ? "後手" : "先手";
        } 
        // 新しく駒を選択する場合
        else {
            // クリックされた駒が現在の手番のプレイヤーのものでない場合は何もせずに終了
            if (this.turn !== this.board[i][j].owner) {
                return;
            }
            
            // 選択状態をセット
            this.selection.isNow = true;
            this.selection.state = true;
            
            // 選択された駒の位置を記録
            this.selection.before_i = i;
            this.selection.before_j = j;
            
            // 盤面の選択情報を初期化（全マスを「未選択」状態に）
            this.selection.boardSelectInfo = JSON.parse(JSON.stringify((new Array(9)).fill((new Array(9)).fill("未選択"))));
            
            // 駒台の選択情報を初期化
            this.selection.pieceStandSelectInfo = {
                "先手": Array(9).fill("未選択"),
                "後手": Array(9).fill("未選択")
            };
            
            // 選択された駒の位置を「選択状態」に設定
            this.selection.boardSelectInfo[i][j] = "選択状態";
            
            // 選択された駒の移動可能な位置をチェック
            this.checkCanPutBoard(i, j);
        }
     
        // 駒を動かした後に詰み判定を行う
       
        await this.checkCheckmate();

        console.log(this.is_check)
        console.log(this.is_check2)

        if (this.is_check2){
            this.undoLastMove(previousState);
            return;  // メソッドを終了
        }
        

        // ゲームが終了した場合、結果を表示
        if (this.isGameOver) {
            alert(`ゲーム終了！ ${this.winner}の勝利です！`);
        }
     }

    // 現在の盤面状態を保存するメソッド
    saveCurrentState() {
        return {
            board: JSON.parse(JSON.stringify(this.board)),
            turn: this.turn,
            pieceStandNum: JSON.parse(JSON.stringify(this.pieceStandNum)),
            pieceStand: JSON.parse(JSON.stringify(this.pieceStand))
        };
    }

    // 最後の手を元に戻すメソッド
    undoLastMove(previousState) {
        this.board = previousState.board;
        this.turn = previousState.turn;
        this.pieceStandNum = previousState.pieceStandNum;
        this.pieceStand = previousState.pieceStand;
        this.selection = new Selection();  // 選択状態をリセット
    }


    //指定された駒が現在の位置から少なくとも1マス移動可能かどうかを判断する
    existCanMove(i, j, piece) {
        // 駒の全ての移動方向についてループ
        for (let l = 0; l < piece.dx.length; l++) {
            // 現在の位置を初期化
            let y = i;
            let x = j;
    
            // 1マス分の移動を計算（先手と後手で移動方向が逆）
            y += this.turn === "先手" ? piece.dy[l] : -piece.dy[l];
            x += this.turn === "先手" ? piece.dx[l] : -piece.dx[l];
    
            // 移動先が盤面内かどうかをチェック
            if (0 <= y && y <= 8 && 0 <= x && x <= 8) {
                // 盤面内に移動可能な場所が一つでもあればtrueを返す
                return true;
            }
        }
    
        // 全ての方向をチェックしても移動可能な場所がなければfalseを返す
        return false;
    }

    checkPromote(piece, i, before_i) {
        if (!piece.getPromotedPiece()) {
            return piece;
        }
        const promoteAreaMinY = piece.owner === "先手" ? 0 : 6;
        const promoteAreaMaxY = piece.owner === "先手" ? 2 : 8;
        if ((promoteAreaMinY <= i && i <= promoteAreaMaxY) || (promoteAreaMinY <= before_i && before_i <= promoteAreaMaxY)) {
            if (window.confirm('成りますか？')) {
                return piece.getPromotedPiece()
            }
        }
        return piece;
    }

    //選択された駒が移動可能な全てのマスを探索し、
    //それらを「配置可能」としてマーク
    async checkCanPutBoard(i, j) {
        // 選択された駒を取得
        const piece = this.board[i][j];
    
        // 駒の全ての移動方向についてループ
        for (let l = 0; l < piece.dx.length; l++) {
            // 現在の位置を初期化
            let y = i;
            let x = j;
    
            // その方向に対して、駒の移動可能マス数だけループ
            for (let _ = 0; _ < piece.dk[l]; _++) {
                // 次のマスの座標を計算（先手と後手で移動方向が逆）
                y += this.turn === "先手" ? piece.dy[l] : -piece.dy[l];
                x += this.turn === "先手" ? piece.dx[l] : -piece.dx[l];
    
                // 盤外に出た場合、または自分の駒がある場合はその方向の探索を終了
                if (y < 0 || y > 8 || x < 0 || x > 8 || this.board[y][x].owner === piece.owner) {
                    break;
                }
    
                // 移動可能なマスとして記録
                this.selection.boardSelectInfo[y][x] = "配置可能";
    
                // 空のマスの場合は次のマスの探索を続行
                if (!this.board[y][x].owner) {
                    continue;
                }
    
                // 相手の駒がある場合はその駒を取れるが、そこで探索終了
                break;
            }
        }
    }

    //持ち駒がクリックされたときに呼び出され、
    //選択状態の更新、盤面の選択可能箇所の更新などを行う
    pieceStandClick(piece) {
        // 既に駒が選択されているか、クリックされた駒が現在の手番のプレイヤーのものでない場合は処理を中断
        if (this.selection.state || this.turn !== piece.owner) {
            return;
        }
    
        // 選択状態を「現在選択中」に設定
        this.selection.isNow = true;
    
        // 選択状態を有効に設定
        this.selection.state = true;
    
        // 盤面の選択情報を初期化（全マスを「未選択」状態に）
        // JSON.parse(JSON.stringify(...))で深いコピーを作成し、参照の問題を回避
        this.selection.boardSelectInfo = JSON.parse(JSON.stringify((new Array(9)).fill((new Array(9)).fill("未選択"))));
    
        // 選択された持ち駒を記録
        this.selection.pieceStandPiece = piece;
    
        // 駒台の選択情報を初期化（全ての駒を「未選択」状態に）
        this.selection.pieceStandSelectInfo = {
            "先手": Array(9).fill("未選択"),
            "後手": Array(9).fill("未選択")
        };
    
        // 選択された駒の駒台上のインデックスを取得
        const i = this.pieceStand[piece.owner].findIndex(p => p.name === piece.name);
    
        // 選択された駒の状態を「選択状態」に更新
        this.selection.pieceStandSelectInfo[this.turn][i] = "選択状態";
    
        // 選択された持ち駒を盤面のどこに置けるかをチェック
        this.checkCanPutPieceStand(piece);
    }

    //プレイヤーの持ち駒（駒台）を更新する
    makePieceStand() {
        // 新しい持ち駒リストを初期化
        let myPieceStand = [];
    
        // 現在の手番（先手/後手）のプレイヤーの持ち駒の数を取得
        const myPieceStandNum = this.pieceStandNum[this.turn];
    
        // 各種類の持ち駒についてループ
        for (let name in myPieceStandNum) {
            // その種類の持ち駒が1つ以上ある場合
            if (myPieceStandNum[name] > 0) {
                // 対応する駒オブジェクトを作成し、持ち駒リストに追加
                myPieceStand.push(Piece.getPieceByName(name, this.turn));
            }
        }
    
        // 持ち駒リストの長さが9未満の間、空の駒（Blank）を追加
        while (myPieceStand.length < 9) {
            myPieceStand.push(new Blank());
        }
    
        // 作成した持ち駒リストを現在のプレイヤーの pieceStand プロパティに設定
        this.pieceStand[this.turn] = myPieceStand;
    }

    //持ち駒を盤面に配置できる場所を判定する
    checkCanPutPieceStand(piece) {
        // 各列に歩を置けるかどうかを記録する配列を初期化。最初は全ての列に置けると仮定。
        let pawnColMemo = Array(9).fill(true);
        
        // 配置しようとしている駒が歩の場合、特別な処理を行う
        if (piece.name === "歩") {
            // 盤面の全マスをチェック
            for (let i = 0; i < 9; i++) {
                for (let j = 0; j < 9; j++) {
                    // 既に自分の歩がある列を見つけた場合
                    if (this.board[i][j].name === "歩" && this.board[i][j].owner === piece.owner) {
                        // その列には新たに歩を置けないようにマーク
                        pawnColMemo[j] = false;
                    }
                }
            }
        }
        
        // 盤面の全マスを再度チェックして、駒を置ける場所を探す
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                // 以下の条件を全て満たす場合、そのマスを「配置可能」とマーク
                // 1. そのマスに駒が存在しない
                // 2. その位置に駒を動かすことが可能
                // 3. 歩の場合、その列にまだ自分の歩がない
                if (!this.board[i][j].owner && this.existCanMove(i, j, piece) && pawnColMemo[j]) {
                    this.selection.boardSelectInfo[i][j] = "配置可能";
                }
            }
        }
    }

}

class Selection {
    boardSelectInfo = JSON.parse(JSON.stringify((new Array(9)).fill((new Array(9)).fill(""))));
    isNow = false;
    state = false;
    before_i = null;
    before_j = null;
    pieceStandSelectInfo = {
        "先手": Array(9).fill("持駒"),
        "後手": Array(9).fill("持駒")
    };
    pieceStandPiece = new Blank();
}

export { BoardInfo, Selection };