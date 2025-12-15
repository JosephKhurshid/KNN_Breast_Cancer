import math as math
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
sns.set(style='white', context='notebook')


# data source
data_local_path_source = "./Breast_Cancer.csv"
# names of colums for headers
col_names = ["Age", "Race", "Marital Status", "T Stage", "N Stage", "6th Stage", "Differentiate", "Grade", "A Stage", "Tumor Size", "Estrogen Status", "Progestrone Status", "Regional Node Examined", "Regional Node Positive", "Survival Months", "Status"]
# create a pandas dataframe
breast_cancer_df_original = pd.read_csv(data_local_path_source, names=col_names)

#breast_cancer_df.columns = breast_cancer_df[0]
breast_cancer_df = breast_cancer_df_original[1:]


plt.figure(figsize=(10,10))

'''
# Question a)
(1 point) Data exploration: Identify the categorical and continuous variables. Plot the histogram of each variable 
(i.e., 16 histograms). How are the variables distributed (e.g., unimodal, bimodal, uniform distributions)?
'''
# Function creates a histogram 
def create_histogram(variable_name, x_label, y_label, histogram_title):
  if (variable_name in ["Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive", "Survival Months"]):
    plt.hist(np.asarray(breast_cancer_df[variable_name], float))
  else:
    plt.hist(breast_cancer_df[variable_name])

  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(histogram_title)
  plt.show()

# Call create_histogram for every variable
for variable_name in col_names:
 create_histogram(variable_name, variable_name, "Amount", "Breast Cancer " + variable_name + " Variable Histogram")



#Question b: Scatterplots of continous variables and survival months
'''
Question b) Plot scatter plots between each continuous feature and the `Survival months' outcome. Following that, 
compute the Pearson's correlation between each continuous feature and the survival months outcome. What associations
do you observe between features and outcome?
'''
def create_scatterplot (data_variable_name, x_label, y_label, scatterplot_title,):
  plt.scatter(np.asarray(breast_cancer_df[data_variable_name], float), np.asarray(breast_cancer_df["Survival Months"], float))
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(scatterplot_title)
  plt.show()

# for variable_name in col_names:
#  if (variable_name in ["Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive"]):
#    create_scatterplot(variable_name, variable_name, "Survival Months", "Breast Cancer " + variable_name + " Variable and Survival Months Scatter Plot")


# for variable_name in col_names:
#  if (variable_name in ["Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive"]):
#    new_df_for_pearson_correlation = breast_cancer_df[["Survival Months", variable_name]].copy()
#    correlation = new_df_for_pearson_correlation.corr('pearson')
#    print(correlation)
#    print('\n')


'''
Question c) Data exploration: Plot grouped bar charts for each categorical feature and
the `Status' outcome. Each cluster of bars should correspond to one value of the categorical
feature. Within each cluster, there should be two bars { one bar corresponding to the `alive'
and one bar corresponding to the `dead' outcome.
'''

def create_bar_chart(categorical_variable):
  # plot and add annotations
  p = sns.countplot(data=breast_cancer_df, x=categorical_variable, hue='Status')
  p.legend(title='Status', bbox_to_anchor=(1, 1), loc='upper left')
  for c in p.containers:
    # set the bar label
    p.bar_label(c, fmt='%.0f', label_type='edge')
  
  plt.show()

# for variable_name in col_names: 
#     if (variable_name not in ["Age", "Tumor Size", "Regional Node Examined", "Regional Node Positive", "Survival Months", "Status"]):
#       create_bar_chart(variable_name)


'''
(d) Classification: Randomly split the data samples into training, validation /
development, and testing sets (e.g., 70-15-15% split). Implement a K-Nearest Neighbor clas-
siffier (K-NN) to classify in terms of patient `Status' (i.e., `alive' / `dead' outcomes). Use the
euclidean distance (l2-norm) as a distance measure for continuous or clearly ordered variables
and the Hamming distance as a distance measure to categorical non-ordered variables (e.g.,
marital status). Explore different values of K = 1; 3; 5; 7; 9; 11; : : :. You will train one model
for each K value using the train data and compute the classification accuracy (Acc), balanced
classification accuracy (BAcc), and F1-score (the latter calculated based on the `dead' class)
(F1) of the model on the validation set. Plot the Acc, BAcc, and F1-score metrics on the
validation set against the different values of K. Please report the best hyper-parameter K*
based on the Acc metric, the best hyper-parameter K** based on the BAcc metric, and the
best hyper-parameter K+ based on the F1 metric. Finally, report the Acc, BAcc, and F1 met-
rics on the test set using K*, K**, and K+.

Please implement the K-NN and the hyper-parameter tuning process. You can use
available libraries that implement basic operations, such as vector/matrix operations, but you
cannot use libraries that implement the K-NN, randomly split the data, or calculate performance
metrics. Please provide your code with comments in the report.
'''

breast_cancer_matrix = breast_cancer_df.to_numpy()

#getting the integer value of the total amount of training samples, dev samples,
# and testing sample under a 70/15/15 % split
def get_total_numbers_for_training_dev_test(data_matrix):

  #get total training samples
  total_training_samples = len(data_matrix)
  #Initialize train/dev/test sample total
  train_sample_total = 0
  dev_sample_total = 0
  test_sample_total = 0

  #if the result of training sample total * .70 is not an integer
  if ((total_training_samples *.7).is_integer() == False):
    #initalize the training sample total number to 70% + 1 to auomatically take care of any decimals.
    train_sample_total = (math.floor(total_training_samples * 0.7)) + 1

    #if the result of training sample total * .15 is not an integer
    if ((total_training_samples *.15).is_integer() == False):
      #initialize the dev sample to 15% + 1
      dev_sample_total = math.floor(total_training_samples *.15) + 1
      #initialize the test sample to 15%
      test_sample_total = math.floor(total_training_samples *.15)
    else: 
      #initialize the dev sample to 15% + 1
      dev_sample_total = total_training_samples *.15
      #initialize the test sample to 15%
      test_sample_total = total_training_samples *.15
  
  #This else statement represents when the result of total_training_samples *.7 is an integer
  else:
    train_sample_total = total_training_samples * 0.7
    if ((total_training_samples *.15).is_integer() == False):
      #initialize the dev sample to 15% + 1
      dev_sample_total = math.floor(total_training_samples *.15) + 1
      #initialize the test sample to 15%
      test_sample_total = math.floor(total_training_samples *.15)
    else: 
      #initialize the dev sample to 15% + 1
      dev_sample_total = total_training_samples *.15
      #initialize the test sample to 15%
      test_sample_total = total_training_samples *.15    

  #if to summation of train/dev/test is greater than total training samples, then subtract one from the dev sample.
  #this might happen due to a simple addition of one to the dev sample when unwarranted 
  if ((train_sample_total+dev_sample_total+test_sample_total) > total_training_samples):
    dev_sample_total = dev_sample_total - 1

  return train_sample_total, dev_sample_total, test_sample_total


# Need to randomize the data, and then split the data into a training, development, and testing set. 
def split_data_into_train_dev_test(data_matrix):

  #randomize data
  np.random.shuffle(data_matrix)

  train_sample_total_number, dev_sample_total_number, test_sample_total_number = get_total_numbers_for_training_dev_test(data_matrix)

  training_data_set = []
  dev_data_set = []
  test_data_set = []

  for sample_index in range(0,len(data_matrix)):
    if (sample_index < train_sample_total_number):
      training_data_set.append(data_matrix[sample_index])
    elif (sample_index < (train_sample_total_number + dev_sample_total_number)):
      dev_data_set.append(data_matrix[sample_index])
    elif (sample_index < train_sample_total_number + dev_sample_total_number + test_sample_total_number):
      test_data_set.append(data_matrix[sample_index])

  return training_data_set, dev_data_set, test_data_set


training_data_set, dev_data_set, test_data_set = split_data_into_train_dev_test(breast_cancer_matrix)

#Euclidean distance calculation on one dimension is p-q squared
def euclidean_distance(input_value, sample_value):
  return ((input_value - sample_value)**2)

#Hamming distance Calculation
def hamming_distance(input_feature_value, sample_feature_value):
  if (input_feature_value == sample_feature_value):
    return 0
  return 1

def is_categorical(feature_index):
  if feature_index in (1, 10,13,14,15):
    return False
  return True


def learn_distance_values_of_input_from_samples(input, set_samples):
  model_numerical_values_over_training_set = []
  for training_sample_index in range(0,len(set_samples)):
    current_sample_value = 0
    for feature_index in range(0, len(set_samples[training_sample_index]) - 1):
      if (is_categorical(feature_index+1)):
        current_sample_value += hamming_distance(input[feature_index],set_samples[training_sample_index][feature_index])
      else: 
        current_sample_value += euclidean_distance(int(input[feature_index]),int(set_samples[training_sample_index][feature_index]))

    model_numerical_values_over_training_set.append(current_sample_value)
  
  ''' Outputs an array with numerical values in place of each training sample. The numerical value represents the comparison between the input and the training sample at that index. 
  The lower the value, the less difference between the input and that sample considered for example: the output would be [0,53,7...], the lowest number there is 0, meaning no differences
  between the input and the training sample at element 0. Thus the status of that training sample is the one we would classify our input as in a 1-Nearest Neighbor algorithm '''
  
  return model_numerical_values_over_training_set

def find_mininum_element_index_value(numerical_distances_from_input_to_samples, skip_index_values):
  
  #The random_index and while loop is to ensure we that we don't get a number already in the skip_index_values and accidentaly
  # create a bug in the code
  random_index = random.randint(1,len(numerical_distances_from_input_to_samples) -1)
  
  while (random_index in skip_index_values):
    random_index = random.randint(1,len(numerical_distances_from_input_to_samples) -1)
  
  min_value = numerical_distances_from_input_to_samples[random_index]
  min_index_value = random_index
  for distance_value_index in range(len(numerical_distances_from_input_to_samples)):
    if distance_value_index not in skip_index_values:
    # If the other element is min than first element
      if numerical_distances_from_input_to_samples[distance_value_index] < min_value: 
        min_value = numerical_distances_from_input_to_samples[distance_value_index] 
        min_index_value = distance_value_index

  return min_index_value

def vote(target_variables):
  dead = 0
  alive = 0
  for variable_value in target_variables: 
    if variable_value == "Alive":
      alive +=1
    else: 
      dead +=1
  
  if dead > alive:
    return "Dead"

  return "Alive"


'''
Simple classification accuraccy (Acc):
( #Correctly Classified Samples for Classified Samples/#Total Samples)
'''
def calculate_simple_accuracy(results_from_knn, dev_set): 
  correctly_classified_samples = float(0)
  for result_index in range(0,len(results_from_knn)): 
    if (results_from_knn[result_index] == dev_set[result_index][15] ):
      correctly_classified_samples +=1
  
  return correctly_classified_samples / float(len(dev_set))

'''
Balanced Classification Accuracy (for unbalanced classes:)
(   (#correctly classified samples for class 1 / #total samples for class1) + ... (#Correctly Classified samples for class K/# total samples for class K)
'''

def calculate_balanced_classification_accuracy(results_from_knn, dev_set):
  correctly_classified_alive_samples = float(0)
  correctly_classified_dead_samples = float(0)
  total_alive_samples = float(0)
  total_dead_samples = float(0)

  for result_index in range(0,len(results_from_knn)):

    if (dev_set[result_index][15] == "Dead"):
      total_dead_samples +=1
    elif (dev_set[result_index][15] == "Alive"):
      total_alive_samples +=1

    if (results_from_knn[result_index] == dev_set[result_index][15]):
      if (results_from_knn[result_index] == "Dead"):
        correctly_classified_dead_samples +=1
      elif (results_from_knn[result_index] == "Alive"): 
        correctly_classified_alive_samples +=1
  
 
  return ( (0.5 * (correctly_classified_alive_samples/total_alive_samples))  + (0.5 * (correctly_classified_dead_samples/total_dead_samples) ) )


'''
Precision (For Binary Classification)
( #correctly classified samples to class 1 / #all samples classified as class 1)

Recall (For Binary Classification):
( #correctly classified samples to class 1 / #all actual samples from class 1)

F1-Score: 2 * ( (precision*recall) / (precision+recall) )
'''

def calculate_f_one_score(results_from_knn, dev_set):
  precision = float(0)
  recall = float(0)
  correctly_classified_dead_samples = float(0)
  all_samples_classified_as_dead = float(0)
  all_actual_samples_from_class_dead = float(0)

  for result_index in range(0,len(results_from_knn)):

    if (results_from_knn[result_index] == "Dead"):
      all_samples_classified_as_dead +=1
  
    if (dev_set[result_index][15] == "Dead"):
      all_actual_samples_from_class_dead +=1

    if (results_from_knn[result_index] == dev_set[result_index][15]):
      if (results_from_knn[result_index] == "Dead"):
        correctly_classified_dead_samples +=1

  precision = correctly_classified_dead_samples / all_samples_classified_as_dead
  recall = correctly_classified_dead_samples / all_actual_samples_from_class_dead

  f_one_score = 2.0 * ( (precision*recall) / (precision+recall))

  return f_one_score


def calculate_f_one_score_for_each_race(results_from_knn, dev_set):

  only_black_data_set = []
  only_black_results_from_knn = []
  only_white_data_set = []
  only_white_results_from_knn = []
  only_other_data_set = []
  only_other_results_from_knn = []

  for index in range(0,len(dev_set)): 
    if (dev_set[index][1] == "Black"):
      only_black_data_set.append(dev_set[index])
      only_black_results_from_knn.append(results_from_knn[index])
    elif (dev_set[index][1] == "White"):
      only_white_data_set.append(dev_set[index])
      only_white_results_from_knn.append(results_from_knn[index])  
    else:
      only_other_data_set.append(dev_set[index])
      only_other_results_from_knn.append(results_from_knn[index]) 

  black_f_one_score = calculate_f_one_score(only_black_results_from_knn, only_black_data_set)
  white_f_one_score = calculate_f_one_score(only_white_results_from_knn, only_white_data_set)
  other_f_one_score = calculate_f_one_score(only_other_results_from_knn, only_other_data_set)
  return [black_f_one_score, white_f_one_score, other_f_one_score]



def classify_input(input_sample):

  # numerical_distances_from_input_to_sample = learn_distance_values_of_input_from_samples(input_sample, dev_data_set)
  numerical_distances_from_input_to_sample = learn_distance_values_of_input_from_samples(input_sample, training_data_set)
  one_nearest_neighbor_index_value = []
  three_nearest_neighbor_index_values = []
  seven_nearest_neighbor_index_values = []

  #find the 1 nearest neighbor
  one_nearest_neighbor_index_value.append(find_mininum_element_index_value(numerical_distances_from_input_to_sample, []))

  #add the element in the first nearest neighbor index array to the three_nearest_neighbor_index_values array
  three_nearest_neighbor_index_values.append(one_nearest_neighbor_index_value[0])

  #add the 2nd and 3rd nearest neighbor index numbers to the three nn index value array
  three_nearest_neighbor_index_values.append(find_mininum_element_index_value(numerical_distances_from_input_to_sample,three_nearest_neighbor_index_values))
  three_nearest_neighbor_index_values.append(find_mininum_element_index_value(numerical_distances_from_input_to_sample,three_nearest_neighbor_index_values))

  #add the first three nearest neighbors to the seven_nearest neighbor index values array
  seven_nearest_neighbor_index_values.append(three_nearest_neighbor_index_values[0])
  seven_nearest_neighbor_index_values.append(three_nearest_neighbor_index_values[1])
  seven_nearest_neighbor_index_values.append(three_nearest_neighbor_index_values[2])
  #add the 4th through 7th nearest neighbors to the seven_nearest neighbor index values array.
  seven_nearest_neighbor_index_values.append(find_mininum_element_index_value(numerical_distances_from_input_to_sample,seven_nearest_neighbor_index_values))
  seven_nearest_neighbor_index_values.append(find_mininum_element_index_value(numerical_distances_from_input_to_sample,seven_nearest_neighbor_index_values))
  seven_nearest_neighbor_index_values.append(find_mininum_element_index_value(numerical_distances_from_input_to_sample,seven_nearest_neighbor_index_values))
  seven_nearest_neighbor_index_values.append(find_mininum_element_index_value(numerical_distances_from_input_to_sample,seven_nearest_neighbor_index_values))

  one_nn_statuses = []
  three_nn_statuses = []
  seven_nn_statuses = []

  for dev_index in one_nearest_neighbor_index_value:
    # one_nn_statuses.append(dev_data_set[dev_index][15])
    one_nn_statuses.append(training_data_set[dev_index][15])

  for dev_index in three_nearest_neighbor_index_values:
    # three_nn_statuses.append(dev_data_set[dev_index][15])
    three_nn_statuses.append(training_data_set[dev_index][15])

  for dev_index in seven_nearest_neighbor_index_values:
    # seven_nn_statuses.append(dev_data_set[dev_index][15])
    seven_nn_statuses.append(training_data_set[dev_index][15])

  one_nn_classification = vote(one_nn_statuses)
  three_nn_classification = vote(three_nn_statuses)
  seven_nn_classification = vote(seven_nn_statuses)

  return one_nn_classification, three_nn_classification, seven_nn_classification


dev_one_nn_classification_results = []
dev_three_nn_classification_results = []
dev_seven_nn_classification_results = []

for input in dev_data_set:
  one_nn_classification, three_nn_classification, seven_nn_classification = classify_input(input)
  dev_one_nn_classification_results.append(one_nn_classification)
  dev_three_nn_classification_results.append(three_nn_classification)
  dev_seven_nn_classification_results.append(seven_nn_classification)

# one_nn_Acc = calculate_simple_accuracy(dev_one_nn_classification_results, dev_data_set)
# one_nn_Bcc = calculate_balanced_classification_accuracy (dev_one_nn_classification_results, dev_data_set)
# one_nn_f1_score = calculate_f_one_score(dev_one_nn_classification_results, dev_data_set)

# three_nn_Acc = calculate_simple_accuracy(dev_three_nn_classification_results, dev_data_set)
# three_nn_Bcc = calculate_balanced_classification_accuracy (dev_three_nn_classification_results, dev_data_set)
# three_nn_f1_score = calculate_f_one_score(dev_three_nn_classification_results, dev_data_set)

# seven_nn_Acc = calculate_simple_accuracy(dev_seven_nn_classification_results, dev_data_set)
# seven_nn_Bcc = calculate_balanced_classification_accuracy (dev_seven_nn_classification_results, dev_data_set)
# seven_nn_f1_score = calculate_f_one_score(dev_seven_nn_classification_results, dev_data_set)

# print("one_nn_Acc: ", one_nn_Acc)
# print("one_nn_Bcc: ", one_nn_Bcc)
# print("one_nn_f1_score: ", one_nn_f1_score)

# print("three_nn_Acc: ", three_nn_Acc)
# print("three_nn_Bcc: ", three_nn_Bcc)
# print("three_nn_f1_score: ", three_nn_f1_score)

# print("seven_nn_Acc: ", seven_nn_Acc)
# print("seven_nn_Bcc: ", seven_nn_Bcc)
# print("seven_nn_f1_score: ", seven_nn_f1_score)

# plt.plot([1, 3, 7], [one_nn_Acc, three_nn_Acc, seven_nn_Acc], 'ro')
# plt.axis((0, 8, 0, 1.5))
# plt.xlabel("K Number")
# plt.ylabel("Acc")
# plt.show()


# plt.plot([1, 3, 7], [one_nn_Bcc, three_nn_Bcc, seven_nn_Bcc], 'ro')
# plt.axis((0, 8, 0, 1.5))
# plt.xlabel("K Number")
# plt.ylabel("Bcc")
# plt.show()

# plt.plot([1, 3, 7], [one_nn_f1_score, three_nn_f1_score, seven_nn_f1_score], 'ro')
# plt.axis((0, 8, 0, 1.5))
# plt.xlabel("K Number")
# plt.ylabel("F1-Score")
# plt.show()

# test_seven_nn_classification_results = []
# for input in test_data_set:
#   one_nn_classification, three_nn_classification, seven_nn_classification = classify_input(input)
#   test_seven_nn_classification_results.append(seven_nn_classification)

# seven_nn_Acc = calculate_simple_accuracy(test_seven_nn_classification_results, test_data_set)
# seven_nn_Bcc = calculate_balanced_classification_accuracy (test_seven_nn_classification_results, test_data_set)
# seven_nn_f1_score = calculate_f_one_score(test_seven_nn_classification_results, test_data_set)

# print("Test seven_nn_Acc: ", one_nn_Acc)
# print("Test seven_nn_Bcc: ", one_nn_Bcc)
# print("Test seven_nn_f1_score: ", one_nn_f1_score)



seven_nn_f1_scores_for_each_race = calculate_f_one_score_for_each_race(dev_seven_nn_classification_results, dev_data_set)

print("The black f1 score: ", seven_nn_f1_scores_for_each_race[0])
print("The white f1 score: ", seven_nn_f1_scores_for_each_race[1])
print("The other f1 score: ", seven_nn_f1_scores_for_each_race[2])

