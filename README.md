# CS5293SP22 - Project 3

# The Unredactor

## Libraries used

- pandas
- nltk
- sklearn

## Assumptions

- Application must be run on a machine with internet connectivity. Core functionality is dependent on fetching data from the git repository.
- Assuming that the unredactor.tsv file has bad lines or corrupted data, such lines are skipped while reading the file
- Given the limited dataset and the quality of the data, the accuracy of the model is very low.

**Note:** Validation data/records are not being used as RandomForestClassifier is used for training and prediction. The model is not improved upon. 

## Functionality


### fetch_data

_Input: None_

_Output: Dataframe containing the data from the tsv file_

This function uses the raw url of the *unredactor.tsv* file from the git repository to read the current data in the file.
Data is read using pandas library and the dataframe is returned

### clean_data

_Input: Dataframe with data from tsv file_

_Output: Adjusted data frame with headers. Sentences are converted to lower case and lemmatized_

This method sets the header for the data frame and loops through all the sentences in the dataframe.
All the sentences are converted to lower case. 
A lemmatizer from nltk library is used to lemmatize the sentences and the updated dataframe is returned

### setup_training_data

_Input: Dataframe with clean data_

_Output: Dataframes containing rows for training and testing (**VALIDATION ROWS EXCLUDED**)_

This method selects the training and testing data from the dataframe. 
The data is stored in two different dataframes and returned

### train_and_predict

_Input: Dataframes with training and testing data_

_Output: Prints 10 predicted names and returns precision, recall and F1 scores_

Sentences from the training data is vectorized using a TF-IDF vectorizer and set as the X. 
The redacted names from the data are set as Y.

A RandomForestClassifier is then initialized with a maximum depth of 70 and trained using X,Y.

*Performing Prediction:*
Inorder to match the number of features, we use vocabulary from the initial vectorizer to create a new vectorizer.
This vectorizer is then used to vectorize the sentences from the training dataframe and fed as input to the model to predict the names.

The first 10 names from the prediction are then copied to an array and printed on the console.
The predictions and actual names from the testing data are compared to acquire the precision, recall, f1 scores and returned as output. 

The main method calls the above functions in a sequence and the resultant scores are printed on the console as output.

## Test Cases

**_test_fetch_data_**

This test case runs the fetch_data method and checks if the returned dataframe is not empty

**_test_setup_training_data_**

Data is fetched, cleaned and setup using the fetch_data, clean_data and setup_training_data functions.
The dataframes returned are checked if they contain data or not.

**_test_train_and_predict_**

This test case runs the complete project (sequentially calls all the functions), then checks if the scores returned as outputs are less than 1 or not. 


## Steps for local deployment

Clone the repository using the command 

`git clone https://github.com/SSharath-Kumar/cs5293sp22-project3`

Install the required dependencies using the command

`pipenv install`

## Running the project

The project is run using the command

`pipenv run python unredactor.py`

## Running the test cases

Test cases can be run using the below command

`pipenv run python -m pytest`