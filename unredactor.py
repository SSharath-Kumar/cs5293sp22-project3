import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.ensemble import RandomForestClassifier

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


# Function to get the latest data from Git repo
# Inputs - None
# Output - Dataframe containing the data from the tsv file
def fetch_data():
    file_url = r'https://raw.githubusercontent.com/cegme/cs5293sp22/main/unredactor.tsv'
    data_frame = pd.read_csv(file_url, sep='\t', on_bad_lines='skip', header=None, quotechar=None, quoting=3)
    return data_frame


# Function to clean the data from the tsv file
# Input - Dataframe with data from tsv file
# Output - Adjusted data frame with headers. Sentences are converted to lower case and lemmatized
def clean_data(df):
    # Adjusting the header
    df.columns = ['Git-ID', 'Category', 'Name', 'Sentence']

    # Initializing the lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Update all sentences to lowercase and lemmatize sentence too
    for i in range(len(df['Sentence'])):
        sen = df['Sentence'].loc[i]
        if type(sen) == str:
            ls = sen.lower()
            ls = lemmatizer.lemmatize(ls)
            df['Sentence'].loc[i] = ls
    return df


# Function to select records for training the model
# Input - Dataframe with clean data
# Output - Dataframes containing rows for training and testing (VALIDATION ROWS EXCLUDED)
def setup_training_data(df):
    training_df = df.loc[df['Category'] == 'training']
    test_df = df.loc[df['Category'] == 'testing']
    return training_df, test_df


# Function to create and train a model to predict names
# Input - Dataframes with training and testing data
# Output - Accuracy Scores
def train_and_predict(train_df, test_df):
    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), analyzer='word')

    # Setting up training data and labels for the model
    training_sentences = train_df['Sentence']
    x_train = vectorizer.fit_transform(training_sentences)
    y_train = train_df['Name']

    # Initialize and train the model
    clf = RandomForestClassifier(max_depth=90)
    clf.fit(x_train, y_train)

    # Create a vectorizer using previous vectorizer's vocabulary
    vectorizer_test = TfidfVectorizer(stop_words='english', vocabulary=vectorizer.vocabulary_)
    test_sentences = test_df['Sentence']
    x_test_vector = vectorizer_test.fit_transform(test_sentences).toarray()

    # Performing predicition
    y_pred = clf.predict(x_test_vector)

    # Copy first 10 predicted names to a list and print
    pred_names = y_pred[:10].tolist()
    print('Few predicted names: ', pred_names)

    validation_names = test_df['Name']
    validation_names = validation_names.values.tolist()

    ps = precision_score(y_pred, validation_names, average='macro', zero_division=0)
    rs = recall_score(y_pred, validation_names, average='macro', zero_division=0)
    fs = f1_score(y_pred, validation_names, average='macro')

    return ps, rs, fs


if __name__ == "__main__":
    data = fetch_data()
    c_data = clean_data(data)
    training_data, testing_data = setup_training_data(c_data)
    p_score, r_score, f_score = train_and_predict(training_data, testing_data)

    print('Precision Score :', p_score)
    print('Recall Score: ', r_score)
    print('F1 - Score: ', f_score)
