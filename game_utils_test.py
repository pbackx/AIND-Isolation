import unittest

import isolation
from game_utils import reachable_spaces


class ReachableSpacesTest(unittest.TestCase):
    def test_empty_board(self):
        player1 = object()
        player2 = object()
        board = isolation.Board(player1, player2, 5, 5)
        number_of_reachable_spaces = len(reachable_spaces(board, player1, 1))

        self.assertEqual(number_of_reachable_spaces, 5*5)