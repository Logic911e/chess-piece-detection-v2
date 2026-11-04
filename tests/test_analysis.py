"""Tests for analysis.py. The engine test runs only where Stockfish is installed."""

from __future__ import annotations

import pytest

from analysis import analyse, fen_problems, find_engine, format_score, placement_problems

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def test_start_position_has_no_problems():
    assert placement_problems(START) == []


def test_missing_kings_are_reported():
    assert placement_problems("8/8/8/4p3/4P3/8/8/8") == ["White has no king.", "Black has no king."]


def test_empty_board_is_reported():
    assert "The board is empty." in placement_problems("8/8/8/8/8/8/8/8")


def test_pawn_on_back_rank_is_reported():
    assert "A pawn stands on the first or last rank." in placement_problems("P3k3/8/8/8/8/8/8/4K3")


def test_too_many_pawns_is_reported():
    problems = placement_problems("4k3/8/8/8/8/PPPPPPPP/P7/4K3")
    assert "White has more than 8 pawns." in problems


def test_placement_check_ignores_whose_turn_it_is():
    # Black to move would be illegal here (White's king is attacked by the rook
    # while White just moved), but placement alone is fine.
    assert placement_problems("4k3/8/8/8/8/8/8/r3K3") == []


def test_full_fen_reports_a_king_in_check_for_the_wrong_side():
    # The side not to move (Black) is in check by the rook: impossible.
    assert fen_problems("4k3/8/8/8/8/8/8/4RK2 w - - 0 1") != []


def test_full_fen_accepts_a_legal_position():
    assert fen_problems("4k3/8/8/8/8/8/8/4K2R w - - 0 1") == []


def test_malformed_fen_raises():
    with pytest.raises(ValueError):
        fen_problems("not a fen")


@pytest.mark.parametrize(
    "kind, value, text",
    [
        ("cp", 35, "+0.35"),
        ("cp", -120, "-1.20"),
        ("cp", 0, "+0.00"),
        ("mate", 3, "#3"),
        ("mate", -2, "#-2"),
    ],
)
def test_format_score(kind, value, text):
    assert format_score(kind, value) == text


def test_find_engine_uses_the_environment_variable(monkeypatch, tmp_path):
    fake = tmp_path / "engine"
    fake.write_text("")
    monkeypatch.setenv("STOCKFISH_PATH", str(fake))
    assert find_engine() == str(fake)
    monkeypatch.setenv("STOCKFISH_PATH", str(tmp_path / "missing"))
    assert find_engine() is None


def test_real_engine_finds_a_mate_in_one():
    engine = find_engine()
    if engine is None:
        pytest.skip("Stockfish is not installed")
    result = analyse(engine, "7k/8/6K1/8/8/8/8/5Q2 w - - 0 1", seconds=0.3)
    assert result["score_kind"] == "mate"
    assert result["score_value"] == 1
    assert result["score_text"] == "#1"
    assert result["best_move_san"].endswith("#")
