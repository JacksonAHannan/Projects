#!/usr/bin/env python3
"""
Tic-Tac-Toe GUI
- Local two-player or single-player vs computer (easy or unbeatable)
- Displays results/scoreboard and lets you play repeatable games
"""
import tkinter as tk
from tkinter import messagebox
import random

# Symbols
HUMAN = 'X'
AI = 'O'

class TicTacToeApp:
    def __init__(self, root):
        self.root = root
        root.title('Tic-Tac-Toe')
        self.mode = tk.StringVar(value='hvc')  # 'hvh' or 'hvc'
        self.difficulty = tk.StringVar(value='unbeatable')  # 'easy' or 'unbeatable'

        # Game state
        self.board = [''] * 9
        self.current_player = HUMAN
        self.game_over = False

        # Scoreboard
        self.score = {'X': 0, 'O': 0, 'Draws': 0}

        # Build UI
        self._build_ui()
        self._update_status("X goes first")

    def _build_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(padx=10, pady=6)

        mode_frame = tk.LabelFrame(top_frame, text='Mode')
        mode_frame.pack(side='left', padx=6)
        tk.Radiobutton(mode_frame, text='Human vs Human', variable=self.mode, value='hvh', command=self.new_game).pack(anchor='w')
        tk.Radiobutton(mode_frame, text='Human vs Computer', variable=self.mode, value='hvc', command=self.new_game).pack(anchor='w')

        diff_frame = tk.LabelFrame(top_frame, text='AI Difficulty')
        diff_frame.pack(side='left', padx=6)
        tk.Radiobutton(diff_frame, text='Easy', variable=self.difficulty, value='easy', command=self.new_game).pack(anchor='w')
        tk.Radiobutton(diff_frame, text='Unbeatable', variable=self.difficulty, value='unbeatable', command=self.new_game).pack(anchor='w')

        control_frame = tk.Frame(top_frame)
        control_frame.pack(side='left', padx=6)
        tk.Button(control_frame, text='New Game', command=self.new_game).pack(fill='x')
        tk.Button(control_frame, text='Reset Scores', command=self.reset_scores).pack(fill='x', pady=(6,0))

        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(padx=10, pady=6)

        self.buttons = []
        for i in range(9):
            b = tk.Button(self.board_frame, text='', font=('Helvetica', 32), width=3, height=1,
                          command=lambda i=i: self.on_click(i))
            b.grid(row=i//3, column=i%3, padx=4, pady=4)
            self.buttons.append(b)

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(padx=10, pady=6, fill='x')

        self.status_label = tk.Label(bottom_frame, text='', anchor='w')
        self.status_label.pack(fill='x')

        score_frame = tk.Frame(self.root)
        score_frame.pack(padx=10, pady=(0,10))
        self.score_label = tk.Label(score_frame, text=self._score_text())
        self.score_label.pack()

    def _score_text(self):
        return f"X: {self.score['X']}    O: {self.score['O']}    Draws: {self.score['Draws']}"

    def _update_status(self, text):
        self.status_label.config(text=text)

    def on_click(self, idx):
        if self.game_over:
            return
        if self.board[idx] != '':
            return

        if self.mode.get() == 'hvc':
            # Human is always X and goes first
            if self.current_player != HUMAN:
                return

        self._make_move(idx, self.current_player)

        winner = self.check_winner(self.board)
        if winner or self.is_full(self.board):
            self._finish_game(winner)
            return

        self._switch_player()

        if self.mode.get() == 'hvc' and self.current_player == AI and not self.game_over:
            # AI move
            self.root.after(150, self.ai_move)

    def _make_move(self, idx, player):
        self.board[idx] = player
        self.buttons[idx].config(text=player)

    def _switch_player(self):
        self.current_player = AI if self.current_player == HUMAN else HUMAN
        self._update_status(f"{self.current_player}'s turn")

    def ai_move(self):
        if self.difficulty.get() == 'easy':
            move = self.random_move()
        else:
            move = self.best_move(self.board, AI)
        if move is not None:
            self._make_move(move, AI)

        winner = self.check_winner(self.board)
        if winner or self.is_full(self.board):
            self._finish_game(winner)
            return
        self._switch_player()

    def random_move(self):
        empties = [i for i, v in enumerate(self.board) if v == '']
        return random.choice(empties) if empties else None

    def new_game(self):
        self.board = [''] * 9
        self.game_over = False
        self.current_player = HUMAN
        for b in self.buttons:
            b.config(text='', bg='SystemButtonFace')
        self._update_status("X goes first")

        # If mode is HVC and AI should go first (not implemented optionally), we keep human X first
        # If you later want toggles to pick who starts, extend UI.

    def reset_scores(self):
        self.score = {'X': 0, 'O': 0, 'Draws': 0}
        self.score_label.config(text=self._score_text())
        self.new_game()

    def _finish_game(self, winner):
        self.game_over = True
        if winner:
            self._highlight_winner(winner)
            message = f"{winner['player']} wins!"
            self.score[winner['player']] += 1
        else:
            message = "It's a draw."
            self.score['Draws'] += 1
        self.score_label.config(text=self._score_text())
        self._update_status(message)
        # Popup
        messagebox.showinfo('Game Over', message)

    def _highlight_winner(self, winner):
        for idx in winner['line']:
            self.buttons[idx].config(bg='lightgreen')

    @staticmethod
    def check_winner(board):
        lines = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)
        ]
        for a, b, c in lines:
            if board[a] and board[a] == board[b] == board[c]:
                return {'player': board[a], 'line': (a, b, c)}
        return None

    @staticmethod
    def is_full(board):
        return all(cell != '' for cell in board)

    # Minimax implementation for unbeatable AI
    def best_move(self, board, player):
        best_score = None
        move = None
        for i in range(9):
            if board[i] == '':
                board[i] = player
                score = self.minimax(board, False, player)
                board[i] = ''
                if best_score is None or score > best_score:
                    best_score = score
                    move = i
        return move

    def minimax(self, board, is_maximizing, ai_player):
        winner = self.check_winner(board)
        if winner:
            if winner['player'] == ai_player:
                return 1
            else:
                return -1
        if self.is_full(board):
            return 0

        if is_maximizing:
            best = -2
            for i in range(9):
                if board[i] == '':
                    board[i] = ai_player
                    val = self.minimax(board, False, ai_player)
                    board[i] = ''
                    best = max(best, val)
            return best
        else:
            opponent = HUMAN if ai_player == AI else AI
            best = 2
            for i in range(9):
                if board[i] == '':
                    board[i] = opponent
                    val = self.minimax(board, True, ai_player)
                    board[i] = ''
                    best = min(best, val)
            return best

if __name__ == '__main__':
    root = tk.Tk()
    app = TicTacToeApp(root)
    root.mainloop()
