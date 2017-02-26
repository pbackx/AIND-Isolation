"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import math

import itertools

from game_utils import reachable_spaces


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def possible_direction_score(game, player):
    """My first try at a score was to count the number of directions that the player can go to.
    I split the board in four quadrants and check to how many quadrants the player can move.

    The results were pretty bad. In hindsight, this is probably to be expected. Instead of counting all
    possible moves, I summarize them into 4 options. So I'm removing information from the score.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    possible_directions = set()
    for move_x, move_y in game.get_legal_moves(player):
        current_x, current_y = game.get_player_location(player)
        if move_x - current_x == 0:
            dir_x = 0
        else:
            dir_x = math.copysign(1, move_x - current_x)
        if move_y - current_y == 0:
            dir_y = 0
        else:
            dir_y = math.copysign(1, move_y - current_y)
        possible_directions.add((dir_x, dir_y))

    return float(len(possible_directions))


def custom_score(game, player):
    """After my failed possible_direction_score attempt, I thought it might be a good idea to actually
    introduce more information into the score. What's the easiest way? Also calculate the number of next
    moves after the current one.

    TODO: other option is to check for partitioning of the board and count the size

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_spaces = len(reachable_spaces(game, player, 2))
    opp_spaces = len(reachable_spaces(game, game.get_opponent(player), 2))
    return float(own_spaces - opp_spaces)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        if game.get_player_location(self) is None:
            return math.ceil(game.height / 2), math.ceil(
                game.width / 2)  # TODO better way of determining an opening move, I should probably also check if I'm the first or second player

        current_best_move = None

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                for iteration in itertools.count(1):
                    current_best_move = self.run_method(game, iteration)
            else:
                current_best_move = self.run_method(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return current_best_move

    def run_method(self, game, depth):
        if self.method == 'minimax':
            _, current_best_move = self.minimax(game, depth)
        elif self.method == 'alphabeta':
            _, current_best_move = self.alphabeta(game, depth)
        else:
            raise NotImplementedError("Only minimax and alphabeta method implemeented")
        return current_best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth <= 0:
            return self.score(game, self), (-1, -1)

        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")

        best_move = (-1, -1)
        for move in game.get_legal_moves():
            step_game = game.forecast_move(move)
            step_score, _ = self.minimax(step_game, depth - 1, not maximizing_player)
            if self.is_better_score(step_score, best_score, maximizing_player):
                best_score = step_score
                best_move = move

        return best_score, best_move

    @staticmethod
    def is_better_score(score_to_test, score, maximizing):
        if maximizing:
            return score_to_test > score
        else:
            return score_to_test < score

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth <= 0:
            return self.score(game, self), (-1, -1)

        if maximizing_player:
            best_score = float("-inf")
        else:
            best_score = float("inf")

        best_move = (-1, -1)
        for move in game.get_legal_moves():
            step_game = game.forecast_move(move)
            step_score, _ = self.alphabeta(step_game, depth - 1, alpha, beta, not maximizing_player)

            # if this is a better score, update the current best score and move
            if self.is_better_score(step_score, best_score, maximizing_player):
                best_score = step_score
                best_move = move

            # check if this score breaks the current alpha/beta bounds and we can prune
            if maximizing_player and best_score >= beta:
                return best_score, best_move
            elif not (maximizing_player) and step_score <= alpha:
                return best_score, best_move

            # update the alpha or beta
            if maximizing_player:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)

        return best_score, best_move
