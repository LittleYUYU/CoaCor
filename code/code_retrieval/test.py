from CodeRetrievalCritic import *
j = CrCritic()

code = 'select col0 from tab0 ;'
qt = 'how to identifi dev qa prod environ in oracl'
qb = 'need information identify whether dev qa prod environment tried using sys context get exact information please help'
print(j.get_reward(code, qt, qt, qb))
print(j.get_reward(code, qb, qt, qb))