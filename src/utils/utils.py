# General libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Scikit Learn libraries
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression

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



def compare_models(reg1, reg2, X, y, random_state_list, metric='default'):

  # Initialize scores list for both classifiers
  scores_1 = []
  scores_2 = []
  diff_scores = []

  X['solubility'] = y['solubility']

  #print(random_state_list)


  # Iterate through samples
  for i in range(0, len(random_state_list)):
    
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
    #print("Iteration %2d score difference = %.6f" % (i + 1, score_1 - score_2))  

  print(f"mean_score_1 {np.mean(scores_1)}, std {np.std(scores_1)}")
  print(f"mean_score_2 {np.mean(scores_2)}, std {np.std(scores_2)}")
  print(stats.ttest_rel(scores_1, scores_2))
  a, b = stats.ttest_rel(scores_1, scores_2)
  if b <= 0.05:
    print("P value menor ou igual a 0.05")
  
  return scores_1, scores_2




def plot_score_dist(reg1_scores, reg2_scores, reg1, reg2):
  plt.clf()
  sns.set(color_codes=True)
  sns.set(rc={'figure.figsize': (8,6)})
  plt.title('{} vs {} scores'.format(reg1, reg2), fontsize='25')
  plt.xlabel('Score Values', fontsize='20')
  plt.ylabel('Score Frequency', fontsize='20')
  sns.distplot(reg1_scores, label='{} scores'.format(reg1))
  sns.distplot(reg2_scores, label='{} scores'.format(reg2))

  plt.legend()
  plt.show()


def plot_pred_expected_results(y_test, y_pred):
  r = r_regression(y_pred.reshape(-1, 1), y_test['solubility'].to_numpy().reshape(-1, 1))

  fig = plt.figure(figsize=[10,8])
  ax = plt.axes()
  plt.xlabel('Expected', fontsize=16)
  plt.ylabel('Predicted', fontsize=16)

  # Set tick font size
  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

  plt.scatter(y_test, y_pred, color='salmon')
  plt.plot(list(range(0, 120)), list(range(0, 120)), color='lightsteelblue')

  plt.title(label="Solubility: Predicted vs Expected", fontsize=15)
  
  plt.xlim([0, 120])
  plt.ylim([0, 120])

  plt.figtext(.15, .85, 'r = {:.2f}'.format(r[0]), size=15)
  plt.show()


def compare_confidence_intervals(reg1_scores, reg2_scores, lower=2.5, upper=97.5):
  reg1_lower = np.percentile(a=reg1_scores, q=lower)
  reg1_upper = np.percentile(a=reg1_scores, q=upper)

  reg2_lower = np.percentile(a=reg2_scores, q=lower)
  reg2_upper = np.percentile(a=reg2_scores, q=upper)

  print(reg1_lower)
  print(reg1_upper)

  print(reg2_lower)
  print(reg2_upper)


def read_datasets(x_train_file, x_test_file, y_train_file, y_test_file):
  folder_path = "../data/"

  x_train = pd.read_csv(folder_path + x_train_file)
  x_test = pd.read_csv(folder_path + x_test_file)
  y_train = pd.read_csv(folder_path + y_train_file)
  y_test = pd.read_csv(folder_path + y_test_file)

  return x_train, x_test, y_train, y_test


def plot_all_compares(score1, score2, score3, score4, score5, score6):
  fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
  fig.suptitle('Comparação do R2 entre modelos durante Bootstrap')

  fig.text(0.5, 0.04, 'Score Values', ha='center')
  fig.text(0.04, 0.5, 'Score Frequency', va='center', rotation='vertical')


  sns.distplot(score1[0], label='{} scores'.format(score1[2]), ax=axes[0,0])
  sns.distplot(score1[1], label='{} scores'.format(score1[3]), ax=axes[0,0])
  axes[0,0].set_title('{} vs {} scores'.format(score1[2], score1[3]))
  # axes[0,0].legend([score1[2], score1[3]])
  axes[0,0].legend()

  sns.distplot(score2[0], label='{} scores'.format(score2[2]), ax=axes[0,1])
  sns.distplot(score2[1], label='{} scores'.format(score2[3]), ax=axes[0,1])
  axes[0,1].set_title('{} vs {} scores'.format(score2[2], score2[3]))
  # axes[0,1].legend([score2[2], score2[3]])
  axes[0,1].legend()

  sns.distplot(score3[0], label='{} scores'.format(score3[2]), ax=axes[0,2])
  sns.distplot(score3[1], label='{} scores'.format(score3[3]), ax=axes[0,2])
  axes[0,2].set_title('{} vs {} scores'.format(score3[2], score3[3]))
  # axes[0,2].legend([score3[2], score3[3]])
  axes[0,2].legend()

  sns.distplot(score4[0], label='{} scores'.format(score4[2]), ax=axes[1,0])
  sns.distplot(score4[1], label='{} scores'.format(score4[3]), ax=axes[1,0])
  axes[1,0].set_title('{} vs {} scores'.format(score4[2], score4[3]))
  # axes[1,0].legend([score4[2], score4[3]])
  axes[1,0].legend()

  sns.distplot(score5[0], label='{} scores'.format(score5[2]), ax=axes[1,1])
  sns.distplot(score5[1], label='{} scores'.format(score5[3]), ax=axes[1,1])
  axes[1,1].set_title('{} vs {} scores'.format(score5[2], score5[3]))
  # axes[1,1].legend([score5[2], score5[3]])
  axes[1,1].legend()

  sns.distplot(score6[0], label='{} scores'.format(score6[2]), ax=axes[1,2])
  sns.distplot(score6[1], label='{} scores'.format(score6[3]), ax=axes[1,2])
  axes[1,2].set_title('{} vs {} scores'.format(score6[2], score6[3]))
  # axes[1,2].legend([score6[2], score6[3]])
  axes[1,2].legend()

  plt.show()