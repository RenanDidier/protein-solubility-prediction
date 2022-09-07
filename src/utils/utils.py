# General libraries
import pandas as pd
import numpy as np

# Scikit Learn libraries
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# Spicy Libraries
import scipy.stats as stats


def kfold(n_splits=10, shuffle=True, random_state=100):
    return KFold(n_splits = n_splits, shuffle = shuffle, random_state = random_state)


def get_adj_r2(n_observations, n_independent_variables, r2_score):
    Adj_r2 = 1 - ((1 - r2_score) * (n_observations - 1)) / (n_observations - n_independent_variables - 1)
    return Adj_r2


def five_two(reg1, reg2, X, y, metric='default'):

  # Choose seeds for each 2-fold iterations
  seeds = [13, 51, 137, 24659, 347]

  # Initialize the score difference for the 1st fold of the 1st iteration 
  p_1_1 = 0.0

  # Initialize a place holder for the variance estimate
  s_sqr = 0.0

  # Initialize scores list for both classifiers
  scores_1 = []
  scores_2 = []
  diff_scores = []

  # Iterate through 5 2-fold CV
  for i_s, seed in enumerate(seeds):

    # Split the dataset in 2 parts with the current seed
    folds = KFold(n_splits=2, shuffle=True, random_state=seed)

    # Initialize score differences
    p_i = np.zeros(2)

    # Go through the current 2 fold
    for i_f, (trn_idx, val_idx) in enumerate(folds.split(X)):
      # Split the data
      trn_x, trn_y = X.iloc[trn_idx], y.iloc[trn_idx]
      val_x, val_y = X.iloc[val_idx], y.iloc[val_idx]

      # Train regression
      reg1.fit(trn_x, trn_y)
      reg2.fit(trn_x, trn_y)

      # Compute scores
      preds_1 = reg1.predict(val_x)
      score_1 = r2_score(val_y, preds_1)
      
      preds_2 = reg2.predict(val_x)
      score_2 = r2_score(val_y, preds_2)

      if metric == "adj_r2":
        score_1 = base_train_adj_r2 = get_adj_r2(
          n_observations=len(trn_y) / 2,
          n_independent_variables=trn_x.shape[1],
          r2_score = score_1
        )

        score_2 = base_train_adj_r2 = get_adj_r2(
          n_observations=len(trn_y) / 2,
          n_independent_variables=trn_x.shape[1],
          r2_score = score_2
        )


      # keep score history for mean and stdev calculation
      scores_1.append(score_1)
      scores_2.append(score_2)
      diff_scores.append(score_1 - score_2)
      print("Fold %2d score difference = %.6f" % (i_f + 1, score_1 - score_2))

      # Compute score difference for current fold  
      p_i[i_f] = score_1 - score_2

      # Keep the score difference of the 1st iteration and 1st fold
      if (i_s == 0) & (i_f == 0):
        p_1_1 = p_i[i_f]

    # Compute mean of scores difference for the current 2-fold CV
    p_i_bar = (p_i[0] + p_i[1]) / 2

    # Compute the variance estimate for the current 2-fold CV
    s_i_sqr = (p_i[0] - p_i_bar) ** 2 + (p_i[1] - p_i_bar) ** 2 

    # Add up to the overall variance
    s_sqr += s_i_sqr
    
  # Compute t value as the first difference divided by the square root of variance estimate
  t_bar = p_1_1 / ((s_sqr / 5) ** .5) 

  print("Regression 1 mean score and stdev : %.6f + %.6f" % (np.mean(scores_1), np.std(scores_1)))
  print("Regression 2 mean score and stdev : %.6f + %.6f" % (np.mean(scores_2), np.std(scores_2)))
  print("Score difference mean + stdev : %.6f + %.6f" 
        % (np.mean(diff_scores), np.std(diff_scores)))
  print("t_value for the current test is %.6f" % t_bar)



def compare_models(reg1, reg2, X, y, metric='default'):

  # Initialize scores list for both classifiers
  scores_1 = []
  scores_2 = []
  diff_scores = []

  X['solubility'] = y['solubility']
  
  random_state_list = [
    4, 10, 33, 42, 22312,
    400, 77, 809, 9, 111
  ]
  
  # Iterate through samples
  for i in range(len(random_state_list)):
    
    sample = X.sample(n=len(X), replace=True, random_state=random_state_list[i])
    #sample.to_csv('~/Desktop/sample{}'.format(i))

    val_y = sample['solubility']
    val_x = sample.loc[:, sample.columns != 'solubility']

    # Compute scores
    preds_1 = reg1.predict(val_x)
    score_1 = r2_score(val_y, preds_1)
    
    preds_2 = reg2.predict(val_x)
    score_2 = r2_score(val_y, preds_2)

    if metric == "adj_r2":
      score_1 = base_train_adj_r2 = get_adj_r2(
        n_observations=len(val_x) / 2,
        n_independent_variables=val_x.shape[1],
        r2_score = score_1
      )

      score_2 = base_train_adj_r2 = get_adj_r2(
        n_observations=len(val_x) / 2,
        n_independent_variables=val_x.shape[1],
        r2_score = score_2
      )


    # keep score history for mean and stdev calculation
    scores_1.append(score_1)
    scores_2.append(score_2)
    diff_scores.append(score_1 - score_2)
    print("Iteration %2d score difference = %.6f" % (i + 1, score_1 - score_2))

  print(f"mean_score_1 {np.mean(scores_1)}, std {np.std(scores_1)}")
  print(f"mean_score_2 {np.mean(scores_2)}, std {np.std(scores_2)}")
  print(stats.ttest_rel(scores_1, scores_2))
  a, b = stats.ttest_rel(scores_1, scores_2)
  if b <= 0.05:
    print("P value menor ou igual a 0.05")
