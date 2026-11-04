"""Position checks and engine analysis, built on python-chess.

Like predict.py and position.py, this module has no FastAPI or config imports.
"""

from __future__ import annotations

import os
import shutil

import chess
import chess.engine

# What is wrong with a position, in words a player would use. Checks that
# depend on whose turn it is are listed separately: they are not a sign of a
# misread photo until the side to move is known.
_STRUCTURE_PROBLEMS = {
    chess.STATUS_EMPTY: "The board is empty.",
    chess.STATUS_NO_WHITE_KING: "White has no king.",
    chess.STATUS_NO_BLACK_KING: "Black has no king.",
    chess.STATUS_TOO_MANY_KINGS: "There is more than one king of one colour.",
    chess.STATUS_TOO_MANY_WHITE_PAWNS: "White has more than 8 pawns.",
    chess.STATUS_TOO_MANY_BLACK_PAWNS: "Black has more than 8 pawns.",
    chess.STATUS_TOO_MANY_WHITE_PIECES: "White has more than 16 pieces.",
    chess.STATUS_TOO_MANY_BLACK_PIECES: "Black has more than 16 pieces.",
    chess.STATUS_PAWNS_ON_BACKRANK: "A pawn stands on the first or last rank.",
}
_CHECK_PROBLEMS = {
    chess.STATUS_OPPOSITE_CHECK: "The side that is not to move is in check.",
    chess.STATUS_TOO_MANY_CHECKERS: "A king is attacked by more pieces than is possible.",
    chess.STATUS_IMPOSSIBLE_CHECK: "A king is in check in a way no legal game can reach.",
}


def _problems(board: chess.Board, include_checks: bool) -> list[str]:
    status = board.status()
    wanted = dict(_STRUCTURE_PROBLEMS)
    if include_checks:
        wanted.update(_CHECK_PROBLEMS)
    return [message for flag, message in wanted.items() if status & flag]


def placement_problems(placement: str) -> list[str]:
    """Reasons a piece placement cannot be a real position (empty list if none).

    Only checks that do not depend on whose turn it is, so a wrong side to
    move is not reported as a misread photo."""
    return _problems(chess.Board(f"{placement} w - - 0 1"), include_checks=False)


def fen_problems(fen: str) -> list[str]:
    """Reasons a full FEN cannot be analysed. Raises ValueError if it cannot be parsed."""
    board = chess.Board(fen)  # ValueError on malformed input
    return _problems(board, include_checks=True)


def find_engine() -> str | None:
    """Path of the UCI engine to use: $STOCKFISH_PATH, else `stockfish` on PATH."""
    configured = os.environ.get("STOCKFISH_PATH")
    if configured:
        return configured if os.path.isfile(configured) else None
    return shutil.which("stockfish")


def format_score(kind: str, value: int) -> str:
    """'+0.35' for centipawns (White's point of view), '#3' or '#-3' for mate."""
    if kind == "mate":
        return f"#{value}"
    return f"{value / 100:+.2f}"


def analyse(engine_path: str, fen: str, seconds: float = 1.0, plies: int = 6) -> dict:
    """Run the engine on a legal position for a fixed time.

    The score is from White's point of view, as on Lichess. Starts a fresh
    engine process per call: simple, and the cost is small next to the
    thinking time."""
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        info = engine.analyse(board, chess.engine.Limit(time=seconds))
        engine_name = engine.id.get("name", "UCI engine")

    score = info["score"].white()
    mate = score.mate()
    if mate is not None:
        kind, value = "mate", mate
    else:
        kind, value = "cp", score.score() or 0
    pv = list(info.get("pv", []))
    best = pv[0] if pv else None
    return {
        "engine": engine_name,
        "fen": board.fen(),
        "score_kind": kind,
        "score_value": value,
        "score_text": format_score(kind, value),
        "best_move_uci": best.uci() if best else None,
        "best_move_san": board.san(best) if best else None,
        "line": board.variation_san(pv[:plies]) if pv else "",
        "depth": int(info.get("depth", 0)),
    }
