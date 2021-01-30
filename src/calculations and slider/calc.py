import random
from matplotlib.pyplot import imshow
import numpy as np


class Computer(object):
  """
  docstring
  """
  def __init__(self, g, h_g, h_i, c_h, c_o):
    for k, v in locals().items():
      if k != 'self':
        if not Computer._valid_prob(v):
          raise ValueError("Value {} is not a valid probability".format(k))
    self._init_probs = {'g': g, 'h_g': h_g, 'h_i': h_i, 'c_h': c_h, 'c_o': c_o}
    self._probs = {'g': g, 'h_g': h_g, 'h_i': h_i, 'c_h': c_h, 'c_o': c_o}
  
  def update(self, g, h_g, h_i, c_h, c_o):
    self._probs = {'g': g, 'h_g': h_g, 'h_i': h_i, 'c_h': c_h, 'c_o': c_o}

  """
  def update(self, char_code, new_prob):
    if char_code not in self._probs:
      raise ValueError("Character code not valid")
    if not Computer._valid_prob(new_prob):
      raise ValueError("Probability has to be between 0 and 1")
    self._probs[char_code] = new_prob
  """



  def _prob_tp(self):
    return self._get_prob('chg') + self._get_prob('wog')

  def _prob_fp(self):
    return self._get_prob('chi') + self._get_prob('woi')

  def _prob_tn(self):
    return self._get_prob('coi') + self._get_prob('whi')

  def _prob_fn(self):
    return self._get_prob('cog') + self._get_prob('whg')

  def _prec_hate(self):
    try:
      return self._prob_tp()/(self._prob_tp() + self._prob_fp())
    except ZeroDivisionError:
      return 0

  def _rec_hate(self):
    try:
      return self._prob_tp()/(self._prob_tp() + self._prob_fn())
    except ZeroDivisionError:
      return 0

  def _prec_off(self):
    try:
      return self._prob_tn()/(self._prob_tn() + self._prob_fn())
    except ZeroDivisionError:
      return 0

  def _rec_off(self):
    try:
      return self._prob_tn()/(self._prob_tn() + self._prob_fp())
    except ZeroDivisionError:
      return 0
    

  def f1_hate(self):
    try:
      return 2 * self._prec_hate() * self._rec_hate() / (self._prec_hate() + self._rec_hate())
    except ZeroDivisionError:
      return 0

  def f1_off(self):
    try:
      return 2 * self._prec_off() * self._rec_off() / (self._prec_off() + self._rec_off())
    except ZeroDivisionError:
      return 0

  def macro_avg_f1(self):
    return (self.f1_hate() + self.f1_off())/2

  def _get_prob(self, code):

    assert code[0] == 'c' or code[0] == 'w'
    assert code[1] == 'h' or code[1] == 'o'
    assert code[2] == 'g' or code[2] == 'i'
    
    prob_1 = self._probs['g'] if code[2] == 'g' else 1 - self._probs['g']
    prob_2 = self._probs['h_' + code[2]] if code[1] == 'h' else 1 - self._probs['h_' + code[2]]
    prob_3 = self._probs['c_' + code[1]] if code[0] == 'c' else 1 - self._probs['c_' + code[1]]

    return prob_1*prob_2*prob_3


  def values_from_labels(self, labels):
    values = []
    for label in labels:
      values.append(self._str_to_value(label))
    return values
  
  def _str_to_value(self, string):
    d = {'Prec Hate': self._prec_hate(), 'Rec Hate': self._rec_hate(), 'F1 Hate': self.f1_hate(),
    'Prec Off': self._prec_off(), 'Rec Off': self._rec_off(), 'F1 Off': self.f1_off(), 'F1 Macro': self.macro_avg_f1()}
    return d[string]

  @staticmethod
  def _valid_prob(prob):
    return 0 <= prob and prob <= 1

def get_prob(code):
    probs = {'g': 'a', 'h_g': 'b', 'h_i': 'c', 'c_h': 'd', 'c_o': 'e'}
    assert code[0] == 'g' or code[0] == 'i'
    assert code[1] == 'h' or code[1] == 'o'
    assert code[2] == 'c' or code[2] == 'w'
    
    
    prob_1 = probs['g'] if code[0] == 'g' else "(1 - {})".format(probs['g'])
    prob_2 = probs['h_' + code[0]] if code[1] == 'h' else "(1 - {})".format(probs['h_' + code[0]])
    prob_3 = probs['c_' + code[1]] if code[2] == 'c' else "(1 - {})".format(probs['c_' + code[1]])

    return "{}*{}*{}".format(prob_1,prob_2,prob_3)

def gen_text():
  g = 'a'
  hg = 'b'
  hi = 'c'
  ch = 'd'
  co = 'e'
  i = '(1 - {})'.format(g)
  og = '(1 - {})'.format(hg)
  oi = '(1 - {})'.format(hi)
  wh = '(1 - {})'.format(ch)
  wo = '(1 - {})'.format(co)

  ghc = get_prob('ghc')
  ihc = get_prob('ihc')
  ioc = get_prob('ioc')
  goc = get_prob('goc')
  gow = get_prob('gow')
  iow = get_prob('iow')
  ihw = get_prob('ihw')
  ghw = get_prob('ghw')

  tp = "({} + {})".format(ghc, gow)
  fp = "({} + {})".format(ihc, iow)
  tn = "({} + {})".format(ioc, ihw)
  fn = "({} + {})".format(goc, ghw)

  prec_hate = "({}/({} + {}))".format(tp, tp, fp)
  rec_hate = "({}/({} + {}))".format(tp, tp, fn)
  prec_off = "({}/({} + {}))".format(tn, tn, fn)
  rec_off = "({}/({} + {}))".format(tn, tn, fp)

  macrof1 = "{}*{}/({} + {}) + {}*{}/({} + {})".format(prec_hate, rec_hate, prec_hate, rec_hate, prec_off, rec_off, prec_off, rec_off)

  print(macrof1)



        

if __name__ == "__main__":
  gen_text()