def reachable_spaces(game, player, depth=3):
    """Calculate the spaces that a player can reach from its current position and given the
    current board state. This does not take into account that the other play may also move and
    block of certain spaces.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    depth : int
        The depth to search

    Returns
    -------
    list<(int, int)>
        A list of spaces the player can reach

    """
    reachable = set()

    if depth == 0:
        return reachable

    for move in game.get_legal_moves(player):
        reachable.add(move)
        game_move_taken = game.forecast_move(move)
        reachable = reachable.union(reachable_spaces(game_move_taken, player, depth - 1))
    return reachable
