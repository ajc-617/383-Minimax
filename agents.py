import random
import math


BOT_NAME =  "Pogchamp"


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
  
    rseed = None  # change this to a value if you want consistent random choices

    def __init__(self):
        if self.rseed is None:
            self.rstate = None
        else:
            random.seed(self.rseed)
            self.rstate = random.getstate()

    def get_move(self, state):
        if self.rstate is not None:
            random.setstate(self.rstate)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move.  Very slow and not always smart."""

    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Gets called by get_move() to determine the value of each successor state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        #minimax will be recursive: base case is when the state is a "full" state: no more states
        successors = state.successors()
        if not successors:
            #return utility of current board
            return state.utility()
        #otherwise more recursion to do
        else:
            #player one's turn, so get the max of the successors
            if state.next_player == 1:
                cur_max = -100000**2
                for cur_succ in successors:
                    if self.minimax(cur_succ) > cur_max:
                        cur_max = self.minimax(cur_succ)
                return cur_max

            #player two's turn, get the minimum of the successors
            elif state.next_player == -1:
                cur_min = 100000**2
                for cur_succ in successors:
                    if self.minimax(cur_succ) < cur_min:
                        cur_min = self.minimax(cur_succ)
                return cur_min


class MinimaxLookaheadAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move.
 
    Hint: Consider what you did for MinimaxAgent. What do you need to change to get what you want? 
    """

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        Gets called by get_move() to determine the value of successor states.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the (possibly estimated) minimax utility value of the state
        """
        if self.depth_limit == 0:
            return self.evaluation(state)
        else:
            return self.minimax_depth(state, self.depth_limit)

    def minimax_depth(self, state, depth):
        """This is just a helper method for minimax(). Feel free to use it or not. """
        successors = state.successors()
        if depth == None:
            if not successors:
                return self.evaluation(state)
        elif depth == 0 or not successors:
            return self.evaluation(state)
        #done with the base cases
        else:
            if state.next_player == 1:
                cur_max = -100000**2
                for cur_succ in successors:
                    if self.minimax_depth(cur_succ, depth-1) > cur_max:
                        cur_max = self.minimax_depth(cur_succ, depth-1)
                return cur_max

            elif state.next_player == -1:
                cur_min = 100000**2
                for cur_succ in successors:
                    if self.minimax_depth(cur_succ, depth-1) < cur_min:
                        cur_min = self.minimax_depth(cur_succ, depth-1)
                return cur_min 


    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        Gets called by minimax() once the depth limit has been reached.  
        N.B.: This method must run in "constant" time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        #for evalutation function, count how many total pieces each character currently has in a row
        #in the columns, rows, and diagonals (while counting duplicates as well). No idea if this evalutation function is good but it's 
        #what I came up with
        columns = state.get_columns
        rows = state.get_rows
        diagonals = state.get_diagonals

        num_pieces_p1 = 0
        num_pieces_p2 = 0

        p1_cur_count = 0
        p2_cur_count = 0

        prev_col, cur_col, prev_row, cur_row, prev_diag, cur_diag = -2
        for col in columns:
            for piece in col:
                #first piece in the column currently
                if prev_col == -2:
                    prev_col == piece
                    continue
                if cur_col == -2:
                    cur_col = piece
                if prev_col == cur_col and cur_col != 0:
                    if cur_col == 1:
                        if p1_cur_count == 0:
                            p1_cur_count += 2
                        else:
                            p1_cur_count += 1
                    else:
                        if p2_cur_count == 0:
                            p2_cur_count += 2
                        else:
                            p2_cur_count += 1
            num_pieces_p1 += p1_cur_count
            num_pieces_p2 += p2_cur_count
            p1_cur_count = 0
            p2_cur_count = 0
            prev_col = -2
            cur_col = -2

        for row in rows:
            for piece in row:
            if prev_row == -2:


        for diag in diagonals:
            for piece in diag:
        #
        # Fill this in!
        #

        # Note: This cannot be "return state.utility() + c", where c is a constant. 
        return 3  # Change this line!


class AltMinimaxLookaheadAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        #
        # Fill this in, if it pleases you.
        #
        return 19  # Change this line, unless you have something better to do.


class MinimaxPruneAgent(MinimaxAgent):
    """Computer agent that uses minimax with alpha-beta pruning to select the best move.
    
    Hint: Consider what you did for MinimaxAgent.  What do you need to change to prune a
    branch of the state space? 
    """
    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not have a depth limit.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to column 1 (we're trading optimality for gradeability here).

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        #
        # Fill this in!
        #
        self.alphabeta(state, -100000**2, 1000000**2)  # Change this line!

    def alphabeta(self, state,alpha, beta):
        """This is just a helper method for minimax(). Feel free to use it or not."""
        #alpha is the minimum value for the node, beta is the max value for the node (for it to possibly be passe dall the way up the game tree)

        successors = state.successors()
        if not successors:
            return state.utility()
        #at max
        if state.next_player() == 1:
            cur_alpha = None
            for cur_succ in state.successors():
                if cur_alpha == None:
                    cur_alpha = self.alphabeta(cur_succ, alpha, beta)
                    continue
                elif beta < cur_alpha:
                    break
                else: 
                    new_alpha = self.alphabeta(state, cur_alpha, beta)
                    if new_alpha > cur_alpha:
                        cur_alpha = new_alpha
            return cur_alpha

        #at min
        else:
            cur_beta = None
            for cur_succ in state.successors():
                if cur_beta == None:
                    cur_beta = self.alphabeta(cur_succ, alpha, beta)
                    continue
                elif alpha > cur_beta:
                    break
                else: 
                    new_beta = self.alphabeta(state, alpha, cur_beta)
                    if new_beta < cur_beta:
                        cur_beta = new_beta
            return cur_beta


def get_agent(tag):
    if tag == 'random':
        return RandomAgent()
    elif tag == 'human':
        return HumanAgent()
    elif tag == 'mini':
        return MinimaxAgent()
    elif tag == 'prune':
        return MinimaxPruneAgent()
    elif tag.startswith('look'):
        depth = int(tag[4:])
        return MinimaxLookaheadAgent(depth)
    elif tag.startswith('alt'):
        depth = int(tag[3:])
        return AltMinimaxLookaheadAgent(depth)
    else:
        raise ValueError("bad agent tag: '{}'".format(tag))       
