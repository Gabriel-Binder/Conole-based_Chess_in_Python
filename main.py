import re
import sys
from typing import Any

is_white = True
board = [
    ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
    ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
    ['O', 'X', 'O', 'X', 'O', 'X', 'O', 'X'],
    ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
    ['O', 'X', 'O', 'X', 'O', 'X', 'O', 'X'],
    ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
    ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
    ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
]

temp_board = board

empty_board = [
    ['O', 'X', 'O', 'X', 'O', 'X', 'O', 'X'],
    ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
    ['O', 'X', 'O', 'X', 'O', 'X', 'O', 'X'],
    ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
    ['O', 'X', 'O', 'X', 'O', 'X', 'O', 'X'],
    ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O'],
    ['O', 'X', 'O', 'X', 'O', 'X', 'O', 'X'],
    ['X', 'O', 'X', 'O', 'X', 'O', 'X', 'O']
]
last_move = None

castling_rights = {
    'white_kingside': True,
    'white_queenside': True,
    'black_kingside': True,
    'black_queenside': True
}

class Piece: #just using classes to make it more readable
    @staticmethod
    def king():
        moves = [
            [-1,1], [0,1], [1,1],
            [-1,0],        [1,0],
            [-1,-1],[0,-1],[1,-1]
        ]
        castle = [[3, 0],[-3,0]]
        collision = True
        return moves, castle, collision

    @staticmethod
    def queen():
        direction = [
            [-1,1], [0,1], [1,1],
            [-1,0],        [1,0],
            [-1,-1],[0,-1],[1,-1]
        ]
        collision = True
        return direction, collision

    @staticmethod
    def knight():
        moves = [
            [-2, 1], [-1, 2], [ 1, 2], [ 2, 1],
            [ 2,-1], [ 1,-2], [-1,-2], [-2,-1]
        ]
        collision = False
        return moves, collision

    @staticmethod
    def bishop():
        direction = [
            [-1, 1], [ 1, 1],
            [-1,-1], [ 1,-1]
        ]
        collision = True
        return direction, collision

    @staticmethod
    def rook():
        direction = [
            [1,0], [-1,0],
            [0,1], [0,-1]
        ]
        castle = [[3,0],[-3,0]]
        collision = True
        return direction, castle, collision

    @staticmethod
    def pawn():
        def white():
            d = [[1, 0]]
            take_d = [[1, -1], [1, 1]]
            return d, take_d

        def black():
            d = [[-1, 0]]
            take_d = [[-1, -1], [-1, 1]]
            return d, take_d

        direction, take_direction = white() if is_white else black()
        collision = True

        return (direction, take_direction), collision

class Moves: #using classes just to make it more readable
    last_move = None
    def __init__(self) -> None:
        self.king = Piece()
        self.queen = Piece()
        self.knight = Piece()
        self.bishop = Piece()
        self.rook = Piece()
        self.pawn = Piece()

    @staticmethod
    def square_to_coords(square: str) -> tuple[int, int]:
        file_to_col = {'a': 0, 'b': 1, 'c': 2, 'd': 3,
                       'e': 4, 'f': 5, 'g': 6, 'h': 7}
        rank = int(square[1])
        file = square[0]
        return 8 - rank, file_to_col[file]  # (row, col)

    @staticmethod
    def scan_directions(to_row, to_col, directions, expected_piece, hint_row=None, hint_col=None):
        candidates = []
        for dr, dc in directions:
            r, c = to_row + dr, to_col + dc
            while 0 <= r < 8 and 0 <= c < 8:
                current_piece = board[r][c]
                if current_piece == expected_piece:
                    if (hint_col is None or hint_col == c) and (hint_row is None or hint_row == r):
                        candidates.append((r, c))
                    break
                elif current_piece != 'X' and current_piece != 'O':
                    break
                r += dr
                c += dc
        return candidates

    @staticmethod
    def scan_move(moves, to_row, to_col, piece):
        candidates = []
        for dr, dc in moves:
            source_row = to_row + dr
            source_col = to_col + dc
            # Make sure we're within bounds
            if 0 <= source_row < 8 and 0 <= source_col < 8:
                expected_piece = piece if is_white else piece.lower()
                if board[source_row][source_col] == expected_piece:
                    candidates.append((source_row, source_col))
        return candidates

    @staticmethod
    def find_origin(move:dict[str, str] | None | dict[str, str | bool | None | Any]):
        if not move or move['type'] != 'move':
            print("Invalid move or move type not 'move'")
            return None
        to_row, to_col = Moves.square_to_coords(move['to'])
        piece_char = move['piece']
        is_capture = move.get('capture', False)
        piece_symbol = piece_char.upper() if is_white else piece_char.lower()
        origin_hint = move.get('origin_hint', '')
        if len(origin_hint) == 2:
            origin_hint_tuple = Moves.square_to_coords(origin_hint)
            if board[origin_hint_tuple[0]][origin_hint_tuple[1]] != piece_symbol:
                print(f"{piece_symbol} could not be found at: {origin_hint}")
                return None
            else:
                return origin_hint_tuple

        hint_row = None
        hint_col = None

        # Interpret the given origin hint
        if len(origin_hint) == 1:
            if origin_hint[0].isalpha():
                hint_col = ord(origin_hint[0].lower()) - ord('a')
            else:
                hint_row = 8 - int(origin_hint[0])

        candidate_origins = []
        if piece_char.upper() == 'P':
            if is_capture:
                if hint_col is not None:
                    if hint_col == to_col:
                        print("Pawns cannot take forwards!")
                        return None
                (direction, take_direction), collision = Piece.pawn()
                if hint_col == None:
                    for tdir in take_direction:
                        if is_white:
                            if board[to_row + tdir[0]][to_col + tdir[1]] == 'P':
                                candidate_origins.append((to_row + tdir[0], to_col + tdir[1]))
                        else:
                            if board[to_row + tdir[0]][to_col + tdir[1]] == 'P':
                                candidate_origins.append((to_row + tdir[0], to_col + tdir[1]))
                else:
                    if is_white:
                        if board[to_row + 1][hint_col] == 'P':
                            candidate_origins.append((to_row + 1, hint_col))
                    else:
                        if board[to_row - 1][hint_col] == 'p':
                            candidate_origins.append((to_row - 1, hint_col))
            else:
                if hint_col is not None:
                    if hint_col != to_col:
                        print("Cannot Move Pawn to diagonal square without capturing!")
                        return None
                if is_white:
                    if to_row == 4:
                        if board[to_row + 1][to_col] == 'P':
                            candidate_origins.append((to_row + 1, to_col))
                        elif board[to_row + 2][to_col] == 'P':
                            candidate_origins.append((to_row + 2, to_col))
                        else:
                            print("White pawn not found for this move")
                            return None
                    else:
                        if board[to_row + 1][to_col] == 'P':
                            candidate_origins.append((to_row + 1, to_col))
                        else:
                            print("White pawn not found for this move")
                            return None
                else:
                    if to_row == 3:
                        if board[to_row - 1][to_col] == 'p':
                            candidate_origins.append((to_row - 1, to_col))
                        elif board[to_row - 2][to_col] == 'p':
                            candidate_origins.append((to_row - 2, to_col))
                        else:
                            print("Black pawn not found for this move")
                            return None
                    else:
                        if board[to_row - 1][to_col] == 'p':
                            candidate_origins.append((to_row - 1, to_col))
                        else:
                            print("Black pawn not found for this move")
                            return None

        elif piece_char.upper() == 'B':
            directions, _ = Piece.bishop()
            expected_piece = 'B' if is_white else 'b'
            candidate_origins = Moves.scan_directions(to_row, to_col, directions, expected_piece, hint_row, hint_col)


        elif piece_char.upper() == 'R':
            directions, _, _ = Piece.rook()
            expected_piece = 'R' if is_white else 'r'
            candidate_origins = Moves.scan_directions(to_row, to_col, directions, expected_piece, hint_row, hint_col)

        elif piece_char.upper() == 'Q':
            directions, _ = Piece.queen()
            expected_piece = 'Q' if is_white else 'q'
            candidate_origins = Moves.scan_directions(to_row, to_col, directions, expected_piece, hint_row, hint_col)

        elif piece_char.upper() == 'N':
            moves, _ = Piece.knight()
            candidate_origins = Moves.scan_move(moves, to_row, to_col, 'N')

        elif piece_char.upper() == 'K':
            moves, castle, collision = Piece.king()
            candidate_origins = Moves.scan_move(moves, to_row, to_col, 'K')

        if len(candidate_origins) != 1:
            if len(candidate_origins) > 1:
                print('Too many pieces can make that move. Please enter Origin Column (eg.: exe3')
            else:
                print('No valid Piece can make that move.')
            return None
        else:
            origin = candidate_origins[0]
        return origin if len(origin) == 2 else None # return as tuple to skip calling Moves.square_to_coords later on

    @staticmethod
    def is_last_move_pawn_double_step():
        if last_move is None:
            return False
        if last_move['piece'] != 'P':
            return False
        origin_row, origin_col = last_move['origin_hint']
        to_square = last_move['to']
        to_row = 8 - int(to_square[1])  # Rank (e.g., 'e4' -> 4 -> row = 8 - 4 = 4)
        # White pawns move up (from row 6 to 4), black pawns move down (from 1 to 3)
        if abs(origin_row - to_row) == 2:
            return True

        return False

    @staticmethod
    def is_king_in_check(board_state) -> bool:
        king_symbol = 'K' if is_white else 'k'
        enemy_color = str.islower if is_white else str.isupper

        # 1. Locate the king
        king_pos = None
        for r in range(8):
            for c in range(8):
                if board_state[r][c] == king_symbol:
                    king_pos = (r, c)
                    break
            if king_pos:
                break
        if not king_pos:
            return True  # Treat as check if king is missing (just to be safe)

        king_row, king_col = king_pos

        # 2. Check threats from sliding pieces (Queen, Rook, Bishop)
        directions_q, _ = Piece.queen()
        directions_r, _, _ = Piece.rook()
        directions_b, _ = Piece.bishop()

        def scan_dirs(dirs, attackers):
            for dr, dc in dirs:
                r, c = king_row + dr, king_col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    piece = board_state[r][c]
                    if piece in ('X', 'O'):
                        r += dr
                        c += dc
                        continue
                    if enemy_color(piece) and piece.lower() in attackers:
                        return True
                    break
            return False

        if scan_dirs(directions_q, ['q']):
            return True
        if scan_dirs(directions_r, ['r']):
            return True
        if scan_dirs(directions_b, ['b']):
            return True

        # 3. Check knight threats
        knight_moves, _ = Piece.knight()
        for dr, dc in knight_moves:
            r, c = king_row + dr, king_col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                piece = board_state[r][c]
                if enemy_color(piece) and piece.lower() == 'n':
                    return True

        # 4. Check pawn threats
        pawn_directions = [[-1, -1], [-1, 1]] if is_white else [[1, -1], [1, 1]]
        for dr, dc in pawn_directions:
            r, c = king_row + dr, king_col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                piece = board_state[r][c]
                if enemy_color(piece) and piece.lower() == 'p':
                    return True

        # 5. Check opposing king
        king_moves, _, _ = Piece.king()
        for dr, dc in king_moves:
            r, c = king_row + dr, king_col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                piece = board_state[r][c]
                if enemy_color(piece) and piece.lower() == 'k':
                    return True

        return False

    @staticmethod
    def is_square_in_check(board_state, square) -> bool:
        enemy_color = str.islower if is_white else str.isupper
        pos = square

        row, col = pos

        # 2. Check threats from sliding pieces (Queen, Rook, Bishop)
        directions_q, _ = Piece.queen()
        directions_r, _, _ = Piece.rook()
        directions_b, _ = Piece.bishop()

        def scan_dirs(dirs, attackers):
            for dr, dc in dirs:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    piece = board_state[r][c]
                    if piece in ('X', 'O'):
                        r += dr
                        c += dc
                        continue
                    if square in attackers:
                        return True
                    break
            return False

        if scan_dirs(directions_q, ['q']):
            return True
        if scan_dirs(directions_r, ['r']):
            return True
        if scan_dirs(directions_b, ['b']):
            return True

        # 3. Check knight threats
        knight_moves, _ = Piece.knight()
        for dr, dc in knight_moves:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8:
                piece = board_state[r][c]
                if enemy_color(piece) and piece.lower() == 'n':
                    return True

            # 4. Check pawn threats
            pawn_directions = [[-1, -1], [-1, 1]] if is_white else [[1, -1], [1, 1]]
            for dr, dc in pawn_directions:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    piece = board_state[r][c]
                    if enemy_color(piece) and piece.lower() == 'p':
                        return True

            # 5. Check opposing king
            king_moves, _, _ = Piece.king()
            for dr, dc in king_moves:
                r, c = row + dr, col + dc
                if 0 <= r < 8 and 0 <= c < 8:
                    piece = board_state[r][c]
                    if enemy_color(piece) and piece.lower() == 'k':
                        return True
        return False

    @staticmethod
    def is_diagonal_clear(from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        if abs(from_row - to_row) != abs(from_col - to_col):
            return False
        row_step = 1 if to_row > from_row else -1
        col_step = 1 if to_col > from_col else -1
        r, c = from_row + row_step, from_col + col_step
        while r != to_row and c != to_col:
            if board[r][c] not in ('X', 'O'):
                return False
            r += row_step
            c += col_step
        return True

    @staticmethod
    def is_straight_clear(from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos
        if from_row == to_row:
            step = 1 if to_col > from_col else -1
            for c in range(from_col + step, to_col, step):
                if board[from_row][c] not in ('X', 'O'):
                    return False
        elif from_col == to_col:
            step = 1 if to_row > from_row else -1
            for r in range(from_row + step, to_row, step):
                if board[r][from_col] not in ('X', 'O'):
                    return False
        else:
            return False  # Not a straight move
        return True

    @staticmethod
    def is_move_valid(move:dict[str, str] | None | dict[str, str | bool | None | Any]) -> bool:
        to_row, to_col = Moves.square_to_coords(move['to'])
        piece = move['piece']
        is_capture = move.get('capture', False)
        target_piece = board[to_row][to_col]
        origin = move['origin_hint']
        is_promotion = move.get('promotion', False)

        # Target square validation. Checks if targets could even be taken
        if is_capture:

            if target_piece in ('X', 'O'):  # Target square empty, so maybe en passant
                # En passant check
                if piece.upper() == 'P':
                    if Moves.is_last_move_pawn_double_step():
                        last_origin_row, last_origin_col = last_move['origin_hint']
                        last_to_square = last_move['to']
                        last_to_row, last_to_col = Moves.square_to_coords(last_to_square)

                        to_row, to_col = Moves.square_to_coords(move['to'])
                        origin_row, origin_col = move['origin_hint']
                        if is_white:
                            # White captures en passant by moving diagonally up-left or up-right
                            if (origin_row == last_to_row) and (abs(origin_col - last_to_col) == 1) and (
                                    to_row == last_to_row - 1) and (to_col == last_to_col):
                                # Valid en passant capture
                                return True
                        else:
                            # Black captures en passant by moving diagonally down-left or down-right
                            if (origin_row == last_to_row) and (abs(origin_col - last_to_col) == 1) and (
                                    to_row == last_to_row + 1) and (to_col == last_to_col):
                                # Valid en passant capture
                                return True

                #regular check
                print("Invalid capture: no piece at target square and not en passant.")
                return False
            if is_white and target_piece.isupper():
                print("Can't capture your own piece.")
                return False
            if not is_white and target_piece.islower():
                print("Can't capture your own piece.")
                return False
        else:
            if target_piece not in ('X', 'O'):
                print("Target square is occupied and move is not a capture.")
                return False #check if target square is emtpy


        if piece.upper() == 'P':
            direction = -1 if is_white else 1
            start_row = 6 if is_white else 1

            if abs(to_row - origin[0]) == 2:
                if origin[0] != start_row:
                    print("Pawns can only move two spaces from their starting square.")
                    return False
                mid_row = origin[0] + direction
                if board[mid_row][origin[1]] not in ('X', 'O'):
                    print("Pawns can't jump over other pieces.")
                    return False

            if move.get("promotion"):
                if (is_white and to_row == 0) or (not is_white and to_row == 7):
                    promoted_piece = move["promotion"].upper() if is_white else move["promotion"].lower()
                    if promoted_piece not in ['Q', 'R', 'B', 'N']:
                        print("Invalid promotion piece.")
                        return False
                    global temp_board
                    temp_board[to_row][to_col] = promoted_piece
                    temp_board[origin[0]][origin[1]] = empty_board[origin[0]][origin[1]] #replace old with empty
                else:
                    print("Promotion only allowed on final rank.")
                    return False
            if to_row == (0 if is_white else 7) and not move.get("promotion"):
                print("Pawns have to promote on last row")
                return False

            if is_white and to_row >= origin[0]:
                print("White pawns cannot move backward.")
                return False
            if not is_white and to_row <= origin[0]:
                print("Black pawns cannot move backward.")
                return False
        elif piece.upper() == 'K':
            from_row, from_col = origin
            row_diff = abs(to_row - from_row)
            col_diff = abs(to_col - from_col)
            if max(row_diff, col_diff) > 1:
                print("Invalid king move: too far.")
                return False
        elif piece.upper() == 'N':
            from_row, from_col = origin
            row_diff = abs(to_row - from_row)
            col_diff = abs(to_col - from_col)
            if (row_diff, col_diff) not in [(1, 2), (2, 1)]:
                print("Invalid knight move.")
                return False
        elif piece.upper() == 'B':
            if not Moves.is_diagonal_clear(origin, (to_row, to_col)):
                print("Bishop path is blocked.")
                return False

        elif piece.upper() == 'R':
            if not Moves.is_straight_clear(origin, (to_row, to_col)):
                print("Rook path is blocked.")
                return False

        elif piece.upper() == 'Q':
            if not (Moves.is_diagonal_clear(origin, (to_row, to_col)) or
                    Moves.is_straight_clear(origin, (to_row, to_col))):
                print("Queen path is blocked.")
                return False

        # Simulate the move
        temp_board = [row[:] for row in board]  # deep copy of the board
        from_row, from_col = origin
        to_row, to_col = Moves.square_to_coords(move['to'])

        # Make the move
        temp_board[to_row][to_col] = temp_board[from_row][from_col]
        temp_board[from_row][from_col] = 'X'

        # En passant capture
        if piece.upper() == 'P' and target_piece in ('X', 'O') and is_capture:
            ep_row = to_row + 1 if is_white else to_row - 1
            temp_board[ep_row][to_col] = 'X'

        # Check if king is in check after move
        if Moves.is_king_in_check(temp_board):
            print("Move would leave your king in check.")
            return False

        return True

    @staticmethod
    def is_castling_valid(move:dict[str, str] | None | dict[str, str | bool | None | Any]) -> bool:
        global temp_board
        if move['type'] == 'castle':
            row = 7 if is_white else 0
            king_col = 4
            if move['side'] == 'king':
                rook_col = 7
                path = [(row, 5), (row, 6)]
                right_key = 'white_kingside' if is_white else 'black_kingside'
            else:
                rook_col = 0
                path = [(row, 3), (row, 2), (row, 1)]
                right_key = 'white_queenside' if is_white else 'black_queenside'

            if not castling_rights[right_key]:
                print("Castling rights have been lost.")
                return False

            if board[row][king_col].lower() != 'k' or board[row][rook_col].lower() != 'r':
                print("King or rook missing.")
                return False

            for r, c in path:
                if board[r][c] not in ('X', 'O'):
                    print("Path blocked for castling.")
                    return False
            for square in (path[0], path[1]):
                if Moves.is_square_in_check(temp_board, square): return False
        return True

    @staticmethod
    def has_valid_move() -> bool:
        own_pieces = ['P', 'K', 'N', 'B', 'R', 'Q'] if is_white else ['p', 'k', 'n', 'b', 'r', 'q']
        #pieces = []
        #for row_idx, row in enumerate(board):
        #    for col_idx, cell in enumerate(row):
        #        search_for = ['P', 'K', 'N', 'B', 'R', 'Q'] if is_white else ['p', 'k', 'n', 'b', 'r', 'q']
        #        if cell in search_for:
        #            pieces.append([cell.upper(), [row_idx, col_idx]])
        #pieces = [[cell.upper(), [row_idx, col_idx]]
        #          for row_idx, row in enumerate(board)
        #          for col_idx, cell in enumerate(row)
        #          if cell in (['P', 'K', 'N', 'B', 'R', 'Q'] if is_white else ['p', 'k', 'n', 'b', 'r', 'q'])] # why am I able to do this????
        pieces = [
            p
            for row_idx, row in enumerate(board)
            for col_idx, cell in enumerate(row)
            if (cell in own_pieces)
            for p in (
                [['R', [row_idx, col_idx]], ['B', [row_idx, col_idx]]] if cell.upper() == 'Q' #replace Q with R and B. No clue if it's actually better
                else [[cell.upper(), [row_idx, col_idx]]] #just the regular stuff
            )
        ] #?????? it got worse lmao
        global temp_board
        def rook_bishop_check(pmoves, pos) -> bool:
            global temp_board # bruh
            lr, lc = pos
            temp_board = [row[:] for row in board] # just a soft copy, but should be good enough
            diagonal = []
            for dlr, dlc in pmoves:
                #check if is pinned by piece not in that diagonal
                for td in [1, -1]:
                    new_lr = lr + td * dlr
                    new_lc = lc + td * dlc
                    if 0 <= new_lr < 8 and 0 <= new_lc < 8 and temp_board[new_lr][new_lc] not in own_pieces:
                        temp_board[new_lr][new_lc] = temp_board[lr][lc]
                        temp_board[lr][lc] = empty_board[lr][lc]
                        if Moves.is_king_in_check(temp_board):
                            return False
                        temp_board = [row[:] for row in board]
                #check the diagonal for valid moves
                for td in [(dlr, dlc), (-dlr, -dlc)]:
                    cr, cc = lr + td[0], lc + td[1]
                    while 0 <= cr < 8 and 0 <= cc < 8:
                        if temp_board[cr][cc] in ('X', 'O'):
                            diagonal.append([cr, cc])
                            cr += td[0]
                            cc += td[1]
                        elif temp_board[cr][cc] in own_pieces:
                            break
                        else:
                            diagonal.append([cr, cc])
                            break
            if not diagonal: return False
            else: return True

        for piece in pieces:
            #if piece[0] == 'P':
            #    print("Pawn")
            #elif piece[0] == 'N':
            #    print("Knight")
            #elif piece[0] == 'K':
            #    print("King")
            #elif piece[0] == 'B':
            #    print("Bishop")
            #elif piece[0] == 'R':
            #    print("Rook")
            #elif piece[0] == 'Q':
            #    print("Queen")
            # Apparently, Python has a "Switch" no
            match piece[0]:
                case 'P':
                    r, c = piece[1]
                    direction = -1 if is_white else 1
                    start_row = 6 if is_white else 1
                    def simulate(m) -> bool:
                        global temp_board
                        # simulate move and check for check
                        temp_board = [row[:] for row in board]
                        temp_board[m[0]][m[1]] = board[r][c]
                        temp_board[r][c] = empty_board[r][c]
                        if not Moves.is_king_in_check(temp_board):
                            return True
                        return False
                    # 1. Forward one square
                    forward_one = (r + direction, c)
                    if 0 <= forward_one[0] < 8:
                        if board[forward_one[0]][forward_one[1]] in ('X', 'O'):  # empty square
                            # simulate move and check for check
                            if simulate(forward_one):
                                return True
                    # cant move forward 2 squares if you cant move forward 1 square. shouldn't need any code for 2 squares

                    # 3. Captures diagonally (left and right)
                    for dc in [-1, 1]:
                        diag = (r + direction, c + dc)
                        if 0 <= diag[0] < 8 and 0 <= diag[1] < 8:
                            target = board[diag[0]][diag[1]]
                            if target not in own_pieces and target not in ('X', 'O'):
                                # enemy piece to capture
                                if simulate(diag):
                                    return True

                    def get_en_passant_target():
                        if not last_move or last_move.get("type") != "move":
                            return None

                        tpiece = last_move["piece"].upper()
                        to_square = last_move["to"]  # like "e5"
                        # Convert to board indices: 'a'->0, 'b'->1,... rank '1'->7 (bottom), '8'->0 (top)
                        file_to_col = lambda f: ord(f) - ord('a')
                        rank_to_row = lambda r: 8 - int(r)

                        to_col = file_to_col(to_square[0])
                        to_row = rank_to_row(to_square[1])

                        # Check if pawn moved 2 squares (white pawn moves from row 6 to 4, black from 1 to 3)
                        if tpiece == 'P':
                            # white pawn double move
                            if last_move["origin_hint"] and last_move["origin_hint"][1] == '2' and to_square[1] == '4':
                                # en passant target is the square behind the pawn
                                return to_row + 1, to_col
                        elif tpiece == 'p':
                            # black pawn double move
                            if last_move["origin_hint"] and last_move["origin_hint"][1] == '7' and to_square[1] == '5':
                                return to_row - 1, to_col

                        return None

                    if get_en_passant_target() is not None:
                        ep_r, ep_c = get_en_passant_target()
                        if ep_r == r + direction and abs(ep_c - c) == 1:
                            temp_board = [row[:] for row in board]
                            # Remove the pawn being captured en passant
                            temp_board[r][ep_c] = empty_board[r][ep_c]
                            # Move your pawn
                            temp_board[ep_r][ep_c] = board[r][c]
                            temp_board[r][c] = empty_board[r][c]
                            if not Moves.is_king_in_check(temp_board):
                                return True

                case 'K':
                    moves, _, _ = Piece.king()
                    r, c = piece[1]
                    for dr, dc in moves:
                        if all(0 <= x < 8 for x in (r + dr, c + dc)): # check if squares actually on the board
                            if board[r + dr][c + dc] not in own_pieces: # check if square occupied by own piece
                                temp_board = [row[:] for row in board]
                                temp_board[r + dr][c + dc] = 'K' if is_white else 'k'
                                temp_board[r][c] = empty_board[r][c]
                                if not Moves.is_king_in_check(temp_board): # check if move would put king in check
                                    temp_board = [row[:] for row in board]
                                    return True
                                else: temp_board = [row[:] for row in board]
                    if is_white:
                        row = 7
                        king_side = 'white_kingside'
                        queen_side = 'white_queenside'
                    else:
                        row = 0
                        king_side = 'black_kingside'
                        queen_side = 'black_queenside'
                    if castling_rights[king_side]:
                        if board[row][5] in ['O', 'X'] and board[row][6] == ['O', 'X']:
                            if not any(Moves.is_square_in_check(board, (row, col)) for col in (4, 5, 6)):
                                return True  # Kingside castling is legal
                    if castling_rights[queen_side]:
                        if board[row][3] == ['O', 'X'] and board[row][2] == ['O', 'X'] and board[row][1] == ['O', 'X']:
                            if not any(Moves.is_square_in_check(board, (row, col)) for col in (4, 3, 2)):
                                return True # Queenside castling is legal

                case 'N':
                    moves, _ = Piece.knight()
                    r, c = piece[1]
                    for mr, mc in moves:
                        if all(0 <= x < 8 for x in (r + mr, c + mc)):  # check if squares actually on the board
                            if board[r + mr][c + mc] not in own_pieces:  # check if square occupied by own piece
                                temp_board[r][c] = empty_board[r][c]
                                if Moves.is_king_in_check(temp_board):
                                    temp_board = [row[:] for row in board]
                                    break
                                else:
                                    temp_board = [row[:] for row in board]
                                    return True

                case 'B':
                    moves = [[1,-1], [1,1]] #don't need to check all direction individually, can just combine two
                    if rook_bishop_check(moves, piece[1]):
                        return True

                case 'R':
                    moves = [[0,1], [1,0]]
                    if rook_bishop_check(moves, piece[1]):
                        return True
        return False

    @staticmethod
    def parse_algebraic(move: str) -> dict[str, str] | None | dict[str, str | bool | None | Any]:
        if move == "Surrender" or Main.is_commands(move):
            return move
        move = move.replace('+', '').replace('#', '')

        if move == "O-O":
            return {"type": "castle", "side": "king"}
        elif move == "O-O-O":
            return {"type": "castle", "side": "queen"}

        promotion = None
        if "=" in move:
            move, promotion = move.split("=")

        # Match complex pattern like Nd3xe5 or Nxe5 or e4 or Rfd1
        pattern = re.compile(
            r'^(?P<piece>[KQRBN])?'  # Optional piece letter
            r'(?P<origin_file>[a-h])?'  # Optional origin file
            r'(?P<origin_rank>[1-8])?'  # Optional origin rank
            r'(x)?'  # Optional capture
            r'(?P<to_file>[a-h])(?P<to_rank>[1-8])$'  # Destination square
        )

        match = pattern.match(move)
        if not match:
            return None

        piece = match.group("piece") or 'P'  # Default to pawn if no letter
        origin_hint = ""
        if match.group("origin_file"):
            origin_hint += match.group("origin_file")
        if match.group("origin_rank"):
            origin_hint += match.group("origin_rank")
        origin_hint = Moves.find_origin({
            "type": "move",
            "piece": piece,
            "to": match.group("to_file") + match.group("to_rank"),
            "capture": 'x' in move,
            "promotion": promotion,
            "origin_hint": origin_hint
        })
        return {
            "type": "move",
            "piece": piece,
            "to": match.group("to_file") + match.group("to_rank"),
            "capture": 'x' in move,
            "promotion": promotion,
            "origin_hint": origin_hint
        }




class Main:
    def __init__(self):
        global is_white
        global last_move
        move_list = []
        surrender = False
        while True:
            is_white = True
            Main.print_board()
            if not Moves.has_valid_move():
                if Moves.is_king_in_check(board):
                    print("Checkmate! Black Wins!\nWhites King is in check and white has no remaining moves!")
                    break
                else:
                    print("Stalemate! \nWhite has no remaining moves and is not in check!")
                    break
            move, move_input = Main.movement()
            while Main.is_commands(move) or move == "Surrender":
                if move == "Surrender":
                    print("White" if is_white else "Black", "Player Surrendered!")
                    print("Black" if is_white else "White", "Wins!")
                    surrender = True
                    break
                elif Main.is_commands(move):
                    Main.command(move)
                    move, move_input = Main.movement()
            if surrender: break;

            Main.apply_move(move)
            last_move = move
            is_white = not is_white
            print(Moves.is_king_in_check(board))
            move_list.append(move_input + ( '#' if not Moves.has_valid_move() and Moves.is_king_in_check(board) else '+' if Moves.is_king_in_check(board) else ''))
            print("Played Moves:", move_list)

            Main.print_board()
            if not Moves.has_valid_move():
                if Moves.is_king_in_check(board):
                    print("Checkmate! White Wins!\nBlacks King is in check and white has no remaining moves!")
                    break
                else:
                    print("Stalemate! \nBlack has no remaining moves and is not in check!")
                    break
            move, move_input = Main.movement()
            while Main.is_commands(move) or move == "Surrender":
                if move == "Surrender":
                    print("White" if is_white else "Black", "Player Surrendered!")
                    print("Black" if is_white else "White", "Wins!")
                    surrender = True
                    break
                elif Main.is_commands(move):
                    Main.command(move)
                    move, move_input = Main.movement()
            if surrender: break;

            Main.apply_move(move)
            last_move = move
            is_white = not is_white
            print(Moves.is_king_in_check(board))
            move_list.append(move_input + ('+' if Moves.is_king_in_check(board) else ''))
            print("Played Moves:", move_list)
        while True:
            after_game = input()
            if Main.is_commands(after_game):
                Main.command(after_game)
            elif after_game == "": print("Enter \"Help\" for List of commands")

    @staticmethod
    def command(input: str):
        if input == "Exit":
            exit()
        elif input == "Restart":
            print("Restarting...")
            exec(open(sys.argv[0]).read())  # Re-run the script
        elif input == "Help":
            print(
                "Commands:\nHelp: Prints this Text\nRestart: Restarts the Script and starts a new game\nExit: Stops the Script")

    @staticmethod
    def movement():
        prompt = "White's Turn: " if is_white else "Black's Turn: "

        while True:
            move_input = input(prompt)
            if Main.is_commands(move_input) or move_input == "Surrender":
                return move_input, None
            move = Moves.parse_algebraic(move_input)

            if move is None:
                print("Invalid notation. \nPlease enter a valid move using FIDE notation or type \"Help\" to get list of commands.")
                continue

            if move['type'] == "move":
                if move['origin_hint'] is None:
                    print("Invalid origin square. See reason above")
                    continue

                if not Moves.is_move_valid(move):
                    print("Invalid move. See reason above.")
                    continue
            elif move['type'] == "castle":
                if not Moves.is_castling_valid(move):
                    print("Invalid castling. See reason above.")
                    continue

            return move, move_input

    @staticmethod
    def apply_move(move):
        global board

        if move['type'] == "move":

            if move['piece'].upper() == 'K':
                if is_white:
                    castling_rights['white_kingside'] = False
                    castling_rights['white_queenside'] = False
                else:
                    castling_rights['black_kingside'] = False
                    castling_rights['black_queenside'] = False
            elif move['piece'].upper() == 'R':
                if is_white and move['origin_hint'][0] == 7:
                    if move['origin_hint'][1] == 0:
                        castling_rights['white_queenside'] = False
                    elif move['origin_hint'][1] == 7:
                        castling_rights['white_kingside'] = False
                elif not is_white and move['origin_hint'][0] == 0:
                    if move['origin_hint'][1] == 0:
                        castling_rights['black_queenside'] = False
                    elif move['origin_hint'][1] == 7:
                        castling_rights['black_kingside'] = False

            from_row, from_col = move['origin_hint']
            to_row, to_col = Moves.square_to_coords(move['to'])
            # Handle en passant removal
            if move['piece'].upper() == 'P' and move.get('capture', False):
                target_piece = board[to_row][to_col]
                print(target_piece)
                if target_piece in ('X', 'O') and Moves.is_last_move_pawn_double_step():
                    ep_row = to_row + 1 if is_white else to_row - 1
                    print(board[ep_row][to_col])
                    board[ep_row][to_col] = empty_board[ep_row][to_col]

            # Move the piece
            board[to_row][to_col] = board[from_row][from_col]
            board[from_row][from_col] = empty_board[from_row][from_col]

            if move.get("promotion"):
                origin = move["origin_hint"]
                promoted_piece = move["promotion"].upper() if is_white else move["promotion"].lower()
                global temp_board
                board[to_row][to_col] = promoted_piece
                board[origin[0]][origin[1]] = empty_board[origin[0]][origin[1]]  # replace old with empty

        elif move['type'] == 'castle':
            row = 7 if is_white else 0
            if move['side'] == 'king':
                # e1 to g1 (white), e8 to g8 (black)
                board[row][6] = board[row][4]  # Move king
                board[row][4] = empty_board[row][4]
                board[row][5] = board[row][7]  # Move rook
                board[row][7] = empty_board[row][7]
            else:
                # e1 to c1 (white), e8 to c8 (black)
                board[row][2] = board[row][4]
                board[row][4] = empty_board[row][4]
                board[row][3] = board[row][0]
                board[row][0] = empty_board[row][0]
            if is_white:
                castling_rights['white_kingside'] = False
                castling_rights['white_queenside'] = False
            else:
                castling_rights['black_kingside'] = False
                castling_rights['black_queenside'] = False
            return  # Exit early — castling doesn't require further handling

    @staticmethod
    def is_commands(input: str) -> bool:
        commands = ['Help', 'Exit', 'Restart']
        if input in commands:
            return True
        else: return False

    @staticmethod
    def print_board() -> None:
        piece_symbols = {
            'k': '♔', 'q': '♕', 'r': '♖', 'b': '♗', 'n': '♘', 'p': '♙',
            'K': '♚', 'Q': '♛', 'R': '♜', 'B': '♝', 'N': '♞', 'P': '♟',
            'O': '\u2003', 'X': '\u2003'
        }
        reset = "\033[0m"
        white_bg = "\033[100m"
        black_bg = "\033[40m"

        # Fullwidth letters a-h
        print("   \uFF41  \uFF42  \uFF43  \uFF44  \uFF45  \uFF46  \uFF47  \uFF49")

        # Map normal numbers 1-8 to fullwidth numbers
        fullwidth_numbers = ["\uFF11", "\uFF12", "\uFF13", "\uFF14", "\uFF15", "\uFF16", "\uFF17", "\uFF18"]

        for r, row in enumerate(board):
            rank_label = fullwidth_numbers[8 - r - 1]  # 8-r mapped to fullwidth
            print(f"{rank_label} ", end="")
            for c, cell in enumerate(row):
                piece = piece_symbols.get(cell, cell)
                bg = white_bg if (r + c) % 2 == 0 else black_bg
                print(f"{bg} {piece} {reset}", end="")
            print(f" {rank_label}")
        print("   \uFF41  \uFF42  \uFF43  \uFF44  \uFF45  \uFF46  \uFF47  \uFF49")


if __name__ == '__main__':
    Main()

