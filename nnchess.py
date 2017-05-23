import chess
import chess.pgn
import chess.svg
from lru import LRU
import numpy as np

# Converter from FEN to bitarray structure
from bitarray import bitarray

# chess databases
import chess.polyglot
import chess.syzygy
from chess.syzygy import MissingTableError

# for move time measurement
from timeit import default_timer as timer
import time


class FENConverter:
    def __init__(self):
        self.space = 12
        self.pieces = { 'p': 0, 'P': 6, 'n': 1, 'N': 7, 'b': 2, 
                       'B': 8, 'r': 3, 'R': 9, 'q': 4, 'Q': 10, 
                       'k': 5, 'K': 11 }

    def beautifyFEN(self, f0):
        f = f0
        newf = []

        for char in f:
            if char.isdigit():
                for i in range(int(char)):
                    newf.append(self.space)
            elif char != '/':
                newf.append(self.pieces[char])

        return newf
    
    def fen2bitarray(self, fen):
        fen = fen.split()
        bfen = self.beautifyFEN(fen[0])

        result = bitarray(8 * 8 * 12) 
        result.setall(False)

        # pieces
        for i in range(64):
            c = bfen[i]
            if c != self.space:
                result[c*64 + i] = 1
        
        # player to move
        if fen[1] == 'w':
            result.append(1) # white move
        else:
            result.append(0) # black move

        # castling rights 
        for i in ['k', 'K', 'Q', 'q']:
            result.append(i in fen[2])
        
        # Cell passed by pond in one move
        # 8 - because we can infer row from the 'player to move'
        result.extend("00000000")
        if fen[3] != '-':
            pos = 8 - (ord(fen[3][0]) - ord('a'))
            result[-pos] = 1
        
        return result


class EarlyGame:
    def __init__(self, fileName):
        self.book = chess.polyglot.open_reader(fileName)
        
    def getMove(self, board):
        try:
            main_entry = self.book.find(board)
            res = main_entry.move()
        except:
            res = None
        
        return res
    

class EndGame:
    def __init__(self, endGameDir):
        self.tablebases = chess.syzygy.open_tablebases(endGameDir)
        
    def evaluate(self, board):
        try:
            # positive - white, negative - black
            return self.tablebases.probe_dtz(board) 
        except:
            return None

    
class TreeTraversal:
  def __init__(self, eg = None, ev_F = None, cache_size = 1000000):
      self.converter = FENConverter()
      self.posCache = LRU(cache_size)
#         engine = chess.uci.popen_engine("./stockfish")
#         engine.uci()
#         self.engine = engine
      self.nodeCnt = 0
      self.ev_F = ev_F
      self.endGame = eg
      self.cache_hits = 0
      self.endgame_evaluations = 0

  UPPERBOUND = 0
  EXACT = 1
  LOWERBOUND = 2
  LOWERBOUND_VALUE = -1000000

  def _trivial(self, t, s):
        return len(t) - len(t.translate(None, s))
    
  def evaluate(self, pos):
      egEval = self.endGame.evaluate(pos)

      if egEval:
        self.endgame_evaluations += 1
        return egEval

      # trivial eval
      if not self.ev_F:
        f = pos.fen()
        f0 = f.split()[0]
        wt = self._trivial(f0, 'P') + self._trivial(f0, 'Q') * 9 + self._trivial(f0, 'N') * 2.9 + \
            self._trivial(f0, 'B') * 3 + self._trivial(f0, 'R') * 5 + self._trivial(f0, 'K') * 1000  
        bl = self._trivial(f0, 'p') + self._trivial(f0, 'q') * 9 + self._trivial(f0, 'n') * 2.9 + \
            self._trivial(f0, 'b') * 3 + self._trivial(f0, 'r') * 5 + self._trivial(f0, 'k') * 1000   
        turn = -1

        if pos.turn:
          turn = 1
          
        return turn * (wt - bl)
      else:
        return self.ev_F(pos)
  
  def search(self, pos, depth, cb = None):
      ans = []
      self.nodeCnt  = 0
      bestScore = self.LOWERBOUND_VALUE - 1
      bestMove = None
      
      player = -1
      
      if pos.turn:
        player = 1

      for m in pos.legal_moves:
          pos.push(m)
          score = -self._negamax(pos, depth - 1, 
            self.LOWERBOUND_VALUE, -self.LOWERBOUND_VALUE, -player)
          ans.append((score, m))
          pos.pop()
          
          if (score > bestScore) or (score == bestScore and np.random.rand() < 0.3):
              bestScore = score
              bestMove = m
              if cb:
                cb(pos, [m])
      
      nodes = self.nodeCnt
      cacheHits = self.cache_hits
      endgame_evaluations = self.endgame_evaluations
      self.endgame_evaluations = 0
      self.nodeCnt = 0
      self.cache_hits = 0
      
      # white should have bigger value first
      return (bestMove, nodes, cacheHits, endgame_evaluations, ans)
          
  def _negamax(self, pos, depth, alpha, beta, turn):
      alphaOrig = alpha
      fen = pos.fen()
  
      cacheHit = self.posCache.has_key(fen)
      
      if cacheHit:
        cacheEntry = self.posCache[fen]

      if cacheHit and cacheEntry[0] >= depth:
          if cacheEntry[1] == self.EXACT:
              self.cache_hits += 1
              return cacheEntry[2]
          elif cacheEntry[1] == self.LOWERBOUND:
              alpha = max(alpha, cacheEntry[2])
          elif cacheEntry[1] == self.UPPERBOUND:
              beta = min(beta, cacheEntry[2])
  
          if alpha >= beta:
              self.cache_hits += 1
              return cacheEntry[2]
      
      self.nodeCnt += 1

      if depth == 0:
          return self.evaluate(pos)
  
      bestValue = self.LOWERBOUND_VALUE
      # optimization
      # childNodes := OrderMoves(childNodes)
      for move in pos.legal_moves:
          pos.push(move)
          v = -self._negamax(pos, depth - 1, -beta, -alpha, -turn)
          pos.pop()
          bestValue = max(bestValue, v)
          alpha = max(alpha, v)
          
          if alpha >= beta:
              break

      flag = self.EXACT
      
      if bestValue <= alphaOrig:
          flag = self.UPPERBOUND
      elif bestValue >= beta:
          flag = self.LOWERBOUND

      cacheEntry = (depth, flag, bestValue)
      self.posCache[fen] = cacheEntry
  
      return bestValue
    

class NNPlayer:
  def __init__(self, openingBookFile, endGameDir, maxDepth, cb = None, ev_F = None, bookMax=-1):
      self.gameStage = 1 # 1 - opening, 2 - midgame, endgame
      self.earlyGame = EarlyGame(openingBookFile)
      self.midGame = TreeTraversal(ev_F = ev_F, eg = EndGame(endGameDir))
      self.bookMax = bookMax
      self.maxDepth = maxDepth
      self.callback = cb
      self.state = {
        "uniq_nodes_calculated": 0, 
        "cache_hits": 0, 
        "time_spent": 0,
        "bookmoves": 0,
        "moveNumber": 1,
        "last_move_eval": None,
        "endgame_evaluations": 0
      }

  def analyze(game):
    res = []
    print game.move_stack
  
  def makeMove(self, board):
      timer_start = timer()

      if self.gameStage == 1:
          m = self.earlyGame.getMove(board)

          if not m:
              self.gameStage = 2
              return self.makeMove(board)
          else:
              self.state["bookmoves"] += 1
          
          if self.bookMax == self.state["moveNumber"]:
              self.gameStage = 2
              
      elif self.gameStage == 2:
          sres = self.midGame.search(board, self.maxDepth, self.callback)
          m = sres[0]
          self.state["uniq_nodes_calculated"] += sres[1]
          self.state["cache_hits"] += sres[2]
          self.state["endgame_evaluations"] += sres[3]
          self.state["last_move_eval"] = sorted(sres[4], key=lambda t: -t[0])[:3]
      
      self.state["time_spent"] = timer() - timer_start
      board.push(m)
      self.state["moveNumber"] += 1


class Simulation:
  def __init__(self, maxDepth1, maxDepth2, eval1 = None, eval2 = None, 
    bookMax1 = -1, bookMax2 = -1, cb = None, **kwargs):
    ## white and black players
    self.p_w = NNPlayer(
      ev_F = eval1, 
      maxDepth = maxDepth1, 
      bookMax = bookMax1,
      cb = cb, **kwargs)
    self.p_b = NNPlayer(
      ev_F = eval2, 
      maxDepth = maxDepth2,
      bookMax = bookMax2,
      cb = cb, **kwargs)
    self.callback = cb
    self.brd = chess.Board() 

  def _iter(self):
    if self.brd.is_game_over():
      return

    if self.brd.turn:
        self.p_w.makeMove(self.brd)
    else:
        self.p_b.makeMove(self.brd)

    if self.callback:
        self.callback(self.brd, [self.brd.peek()])

  def setBoard(self, newBoard):
    self.brd = newBoard.copy()

  def incremental(self):
    self._iter()

    print self.p_w.state
    print self.p_b.state
    print self.brd.fen()

  def start(self, moves):
    while not self.brd.is_game_over() and \
      self.p_b.state["moveNumber"] != moves:
        self._iter()
        