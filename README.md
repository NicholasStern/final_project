# Predicting Stock Trends w/ Reinforcement Learning

## Introduction


## Problem Definition

### State space representation
States are defined to be the upward/downward price trends of a particular stock for the previous "h" recorded days 
on the stock market. Each upward/downward trend is determined by the difference in the opening and closing prices of the 
stock on that day. Upward movement is encoded as 1, downward movement is encoded as 0, and days with no available
information are encoded as -1. An agent might have no information if they are just starting an episode and are unaware of 
any past price information. An example state representation could be the following:

state = (-1, 0, 1)

Note that dates with the same opening and closing prices were encoded as a decrease to limit the number of states.