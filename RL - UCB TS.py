# Reinforcement Learning - UCB & TS
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import random

dataset = pd.read_csv('D:\Programming Tutorials\Machine Learning\Projects\Datasets\Ads_CTR_Optimisation.csv')

# implementing UCB
def upper_confidence_bound():
    N = 10000
    d = 10
    number_of_selections = [0] * d
    sum_of_rewards = [0] * d
    ad_selected = []
    total_reward = 0
    for n in range(N):
        max_upper_bound = 0
        ad = 0
        for i in range(d):
            if number_of_selections[i] > 0:
                avg_reward = sum_of_rewards[i] / number_of_selections[i]
                deltaI = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
                upper_bound = avg_reward + deltaI
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ad_selected.append(ad)
        number_of_selections[ad] += 1
        reward = dataset.values[n, ad]
        sum_of_rewards[ad] += reward
        total_reward += reward    
    plot.figure(figsize = (10, 10))
    plot.hist(ad_selected)
    plot.title('Histogram')
    plot.xlabel('Ads')
    plot.ylabel('No of times')
    plot.show()

upper_confidence_bound()

# implementing Thompson sampling
def thompson_sampling():
    N = 10000
    d = 10
    ad_selected = []
    numbers_of_rewards_1 = [0] * d
    numbers_of_rewards_0 = [0] * d
    total_reward = 0
    for n in range(N):
        ad = 0
        max_random = 0
        for i in range(d):
            random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
            if random_beta > max_random:
                max_random = random_beta
                ad = i
        ad_selected.append(ad)
        reward = dataset.values[n, ad]
        if reward == 1:
            numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
        else:
            numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
        total_reward += reward
    plot.figure(figsize = (10, 10))
    plot.hist(ad_selected)
    plot.title('Histogram')
    plot.xlabel('Ads')
    plot.ylabel('No of times')
    plot.show()

thompson_sampling()
