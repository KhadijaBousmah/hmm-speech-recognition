# Academic Project (final year of bachlor's degree)

This project was carried out during a university internship supervised by my professor in the final year of my bachelor's degree.

It is based on Lawrence Rabiner's article, `A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition`, and aims to understand the main ideas behind Hidden Markov Models (HMMs) through simple Python implementations.

The project is organized around two directions:

1. a pedagogical study of the three classical HMM problems:
- evaluation of an observation sequence
- decoding of the most likely hidden-state sequence
- re-estimation of model parameters

2. small experimental applications:
- a simple yes/no recognition prototype
- a model for classifying student grade evolutions into increasing, decreasing, and stable profiles

The current code includes:
- Gaussian emissions
- the Viterbi algorithm
- the Forward and Backward algorithms
- Gamma and Ksi computations
- a simplified Baum-Welch-style learning loop

The code is written in Python and is intended as an educational prototype rather than a fully optimized speech-recognition system.
