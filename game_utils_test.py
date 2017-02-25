import unittest

import isolation
from game_utils import reachable_spaces


class ReachableSpacesTest(unittest.TestCase):
    def test_empty_board(self):
        player1 = object()
        player2 = object()
        board = isolation.Board(player1, player2, 5, 5)

        self.assertEqual(len(reachable_spaces(board, player1, 1)), 5*5)
        self.assertEqual(len(reachable_spaces(board, player1, 2)), 5*5)

    def test_depth_of_1(self):
        player1 = object()
        player2 = object()
        board = isolation.Board(player1, player2, 5, 5)
        board.apply_move((2,2))
        board.apply_move((3,2))

        self.assertEqual(len(reachable_spaces(board, player1, 1)), 8)
        self.assertEqual(len(reachable_spaces(board, player1, 2)), 16)