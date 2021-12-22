from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, NuSVC
from sklearn.neural_network import MLPClassifier
from Classifier import Classifier
from deepforest import CascadeForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# MNIST DATASET
# https://www.kaggle.com/c/digit-recognizer/data
# MNIST FASHION DATASET
# https://www.kaggle.com/zalando-research/fashionmnist

# models = [
#     ('hgb',HistGradientBoostingClassifier(l2_regularization=1.5, max_depth=25, max_iter=1500, random_state=2019)),
#     ('svc', SVC()),
#     ('xgb', XGBClassifier(n_jobs=-1, verbose=False)),
# ]

# classifiers = [
#     # LogisticRegression(C=109.85411419875572, n_jobs=-1),
#     # ExtraTreesClassifier(n_jobs=-1, random_state=1),
#     # RandomForestClassifier(random_state=42, n_jobs=-1),
#     # LGBMClassifier(),
#     # XGBClassifier(tree_method='hist', n_jobs=-1, verbose=False),
#     # CatBoostClassifier(depth=8, iterations=34, learning_rate=0.5216607686768668),
#     # KNeighborsClassifier(n_jobs=-1),
#     # HistGradientBoostingClassifier(l2_regularization=1.5, max_depth=25, max_iter=1500, random_state=2019),
#     # MLPClassifier(),    
#     # CascadeForestClassifier(random_state=8, n_jobs=-1, verbose=0),
#     # SVC(),
#     # BaggingClassifier(base_estimator=SVC(), n_jobs=-1),
#     # CalibratedClassifierCV(n_jobs=-1),
#     # SGDClassifier(n_jobs=-1),
#     # StackingClassifier(estimators=models, n_jobs=-1, final_estimator=LogisticRegression(n_jobs=-1)),
# ]

for models in classifiers:
    clf = Classifier()
    clf.run(models, True)
