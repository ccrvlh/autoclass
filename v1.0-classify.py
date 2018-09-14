######################################################################################## Remove warnings

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

######################################################################################## Importing Modules

from sklearn.model_selection import train_test_split as tts
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator
from transformer import DFFunctionTransformer
from transformer import DFFeatureUnion
from transformer import DFStandardScaler
import re
from sklearn.model_selection import GridSearchCV


######################################################################################## Loading data & Preprocessing



# Load the dataset - CSV input
data = pd.read_csv('data/class_full.csv',
                    encoding='latin1',
                    error_bad_lines=False,
                    delimiter=';',
                    decimal=',')

# Define column names & Change label to Pandas "Category"
data.columns  = ['desc', 'value', 'label']


# Define features, numeric_features and target(label).
features = [c for c in data.columns.values if c not in ['label']]
numeric_features = [c for c in data.columns.values if c not in ['desc','label']]
target = 'label'


# Split data into training and testing
train_features, test_features, train_labels, test_labels = tts(data[features], data[target], test_size=0.05, random_state=42)
# X_Train, X_Test, Y_Train, Y_Test

# Transformer to select a single column from the data frame to perform additional transformations on.
# Use on text columns in the data
class TextSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


# Transformer to select a single column from the data frame to perform additional transformations on. 
# Use on numeric columns in the data
class NumberSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]



desc = Pipeline([
                ('selector', TextSelector(key='desc')),
                ('tfidf', TfidfVectorizer(stop_words='english'))
            	])


desc.fit_transform(train_features)

value =  Pipeline([
                ('selector', NumberSelector(key='value')),
                # ('standard', StandardScaler())
            	])

value.fit_transform(train_features)

feats = FeatureUnion([
					('desc', desc),
					('value', value)])

feature_processing = Pipeline([('feats', feats)])
feature_processing.fit_transform(train_features)


pipeline = Pipeline([
    ('features',	feats),
    ('classifier', 	SVC(kernel='linear', random_state = 42)),
])

pipeline.fit(train_features, train_labels)

predictions = pipeline.predict(test_features)
print(numpy.mean(predictions == test_labels))
pipeline.get_params().keys()

# hyperparameters = { 'features__desc__tfidf__max_df': [0.9, 0.95],
#                     'features__desc__tfidf__ngram_range': [(1,1), (1,2)],
#                    	'classifier__max_depth': [50, 70],
#                     'classifier__min_samples_leaf': [1,2]
#                   }

# clf = GridSearchCV(pipeline, hyperparameters, cv=5)
 
# # Fit and tune model
# clf.fit(train_features, train_labels)
# clf.best_params_

# #refitting on entire training data using best settings
# clf.refit

# predictions = clf.predict(test_features)
# probs = clf.predict_proba(test_features)

# print(numpy.mean(predictions == test_labels))

# submission = pd.read_csv('data/test.csv')

# #preprocessing
# submission = processing(submission)
# predictions = clf.predict_proba(submission)

# preds = pd.DataFrame(data=predictions, columns=clf.best_estimator_.named_steps['classifier'].classes_)

# #generating a submission file
# result = pd.concat([submission[['id']], preds], axis=1)
# result.set_index('id', inplace = True)
# result.head()






















######################################################################################## Fitting the models



# #Test
# train_features, test_features, train_labels, test_labels = tts(vectors, labels, test_size=0.05)

# # Random Forest Classifier
# print('\nEstimating score with Random Forest Classifier...')
# forest_model = RandomForestClassifier(random_state=42)
# forest_model.fit(train_features, train_labels)
# predictions_forest = forest_model.predict(vectors_)
# print('Score: {}'.format(forest_model.score(test_features, test_labels)) + ' Random Forest Classifier')


# # Decision Tree Classifier
# print('\nEstimating score with Decision Tree Classifier...')
# tree_model = tree.DecisionTreeClassifier(random_state=42)
# tree_model.fit(train_features, train_labels)
# predictions_tree = tree_model.predict(vectors_)
# print('Score: {}'.format(tree_model.score(test_features, test_labels)) + ' Decistion Tree Classifier')


# # SVC Linear Classifier
# print('\nEstimating score with SVC Linear Classifier...')
# svc_model = svm.SVC(kernel='linear', random_state=42, probability=True)
# svc_model.fit(train_features, train_labels)
# predictions_svc = svc_model.predict(vectors_)
# print('Score: {}'.format(svc_model.score(test_features, test_labels)) + ' SVC Linear Classifier')


# # ExtraTree Classifier
# print('\nEstimating score with ExtraTree Classifier...')
# extra_model = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=10, random_state=42)
# extra_model.fit(train_features, train_labels)
# predictions_extra = extra_model.predict(vectors_)
# print('Score: {}'.format(extra_model.score(test_features, test_labels)) + ' ExtraTree Classifier')



# ######################################################################################## Print consolidated results and accuracy



# # Print of all results
# print('\nScore: {}'.format(forest_model.score(test_features, test_labels)) + ' Random Forest Classifier')
# print('Score: {}'.format(tree_model.score(test_features, test_labels))     + ' Decistion Tree Classifier')
# print('Score: {}'.format(svc_model.score(test_features, test_labels))      + ' SVC Linear Classifier')
# print('Score: {}'.format(extra_model.score(test_features, test_labels))    + ' ExtraTree Classifier\n')


# # Voting Classifier
# eclf = VotingClassifier(estimators=[
#                                    ('lr', forest_model),
#                                    ('rf', tree_model),
#                                    ('svc', svc_model),
#                                    ('extra', extra_model)
#                                    ], voting='soft')

# for clf, label in zip(   [forest_model, tree_model, svc_model, extra_model, eclf],
#                          ['Random Forest', 'Decision Tree', 'SVC', 'Extra']):
     
#      scores = cross_val_score(clf, test_features, test_labels, cv=5, scoring='accuracy')
#      print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))



# ######################################################################################## Make Predictions



# # Statement to be classified
# statement = pd.read_csv('data/extrato.csv',
#                     encoding='latin1',
#                     error_bad_lines=False,
#                     delimiter=',')

# # Defining the column and vectorizing it
# newFeatures     = statement['memo'].values.astype(str)
# newVectorizer   = TfidfVectorizer()
# newVector       = newVectorizer.fit_transform(newFeatures)
# newVector_      = newVectorizer.transform(newFeatures)

# # Predictions with Random Forest, Decision Tree & SVC models
# predictions_forest  = forest_model.predict(vectorizer.transform(newFeatures))
# predictions_forest  = numpy.asarray(predictions_forest)
# predictions_forest  = pd.DataFrame(predictions_forest)

# predictions_tree  = tree_model.predict(vectorizer.transform(newFeatures))
# predictions_tree  = numpy.asarray(predictions_tree)
# predictions_tree  = pd.DataFrame(predictions_tree)

# predictions_svc  = svc_model.predict(vectorizer.transform(newFeatures))
# predictions_svc  = numpy.asarray(predictions_svc)
# predictions_svc  = pd.DataFrame(predictions_svc)

# predictions_extra = extra_model.predict(vectorizer.transform(newFeatures))
# predictions_extra  = numpy.asarray(predictions_extra)
# predictions_extra  = pd.DataFrame(predictions_extra)

# # Finding probabilities for each of the assigned categories
# forest_prob = forest_model.predict_proba(vectorizer.transform(newFeatures))
# forest_prob = numpy.asarray(forest_prob)
# forest_prob = pd.DataFrame(forest_prob)
# forest_prob = forest_prob.max(axis=1)

# tree_prob = tree_model.predict_proba(vectorizer.transform(newFeatures))
# tree_prob = numpy.asarray(tree_prob)
# tree_prob = pd.DataFrame(tree_prob)
# tree_prob = tree_prob.max(axis=1)

# svc_prob = svc_model.predict_proba(vectorizer.transform(newFeatures))
# svc_prob = numpy.asarray(svc_prob)
# svc_prob = pd.DataFrame(svc_prob)
# svc_prob = svc_prob.max(axis=1)

# extra_prob = extra_model.predict_proba(vectorizer.transform(newFeatures))
# extra_prob = numpy.asarray(extra_prob)
# extra_prob = pd.DataFrame(extra_prob)
# extra_prob = extra_prob.max(axis=1)



# ######################################################################################## Export and consolidate predictions



# csv_pred = pd.concat([  predictions_forest, forest_prob,
#                     predictions_tree, tree_prob,
#                     predictions_svc, svc_prob,
#                          predictions_extra, extra_prob],
#                     axis=1)

# csv_pred.to_csv("data/predictions.csv")


# consolidated = pd.concat([statement, csv_pred], axis=1)
# consolidated.columns  = ['type', 'date', 'amount', 'memo', 'id',
#                     'forest_predict', 'forest_prob',
#                     'tree_predict', 'tree_prob',
#                     'svc_predict', 'svc_prob',
#                          'extra_predict', 'extra_prob']

# consolidated.to_csv("data/consolidated.csv")



# ######################################################################################## END