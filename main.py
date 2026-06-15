# il faut normaliser les donnees d'entrees 

# the Markov model is basically composed of a A, B and p, where A is the transition matrix, B is the observation matrix and p is the initial state distribution.
# in our case, the initial state is always the same, because we are working with left-right models. 
# the first problem is to compute the probability of an observation sequence given the model.

# I will first try to model a simple Markov model, with 2 states and 20 observations (students grades)
# first state for the fisrt part of the year, then the second state is for the second part of the year
# we want to model 3 types of students : the students who improve, the students who get worse and the students who stay the same.
# we need 3 models for each type of students.

# first model Imp : (A, B) 
# A transition matrix from state 1 to state 2.
# if we are in state 1 we have 25% chance to stay in state 1 and 75% chance to go to state 2.
# and if we are in state 2 we have 0% chance to go, and 25% chance to stay in state 2.
ImpA = [
    [0.25, 0.75],
    [0, 1]
]

# B observation matrix, we have 20 observations, we will model the grades from 0 to 20.
# for the first state, the students have equal chance to get any grade from 0 to 20
ImpB = [ 
    [0.047] * 21,  # 21 grades from 0 to 20
    []     # second part of the year
]




