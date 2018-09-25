import time
from CodeRetrievalCritic import *
j = CrCritic()

s0 = time.time()
code = 'select col0 from tab0 ;'
qt = 'how to identifi dev qa prod environ in oracl'
qb = 'need information identify whether dev qa prod environment tried using sys context get exact information please help'
print(j.get_reward(code, qt, qt, qb, number_of_runs=10))
print(j.get_reward(code, qb, qt, qb, number_of_runs=10))
print("Test time: %.3f" % (time.time() - s0))