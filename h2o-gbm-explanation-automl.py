
# coding: utf-8

# In[1]:


from __future__ import print_function
import sklearn
import sklearn.datasets
import sklearn.ensemble
import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import lime
import lime.lime_tabular
import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.automl import H2OAutoML


# In[2]:


h2o.init(max_mem_size = "500M", nthreads = 2) 
h2o.remove_all()


# In[3]:


class h2o_predict_proba_wrapper:
    # drf is the h2o distributed random forest object, the column_names is the
    # labels of the X values
    def __init__(self,model,column_names):
            
            self.model = model
            self.column_names = column_names
 
    def predict_proba(self,this_array):        
        # If we have just 1 row of data we need to reshape it
        shape_tuple = np.shape(this_array)        
        if len(shape_tuple) == 1:
            this_array = this_array.reshape(1, -1)
            
        # We convert the numpy array that Lime sends to a pandas dataframe and
        # convert the pandas dataframe to an h2o frame
        self.pandas_df = pd.DataFrame(data = this_array,columns = self.column_names)
        self.h2o_df = h2o.H2OFrame(self.pandas_df)
        
        # Predict with the h2o drf
        self.predictions = self.model.predict(self.h2o_df).as_data_frame()
        # the first column is the class labels, the rest are probabilities for
        # each class
        self.predictions = self.predictions.iloc[:,1:].as_matrix()
        return self.predictions
    
    def partial_dep_plot(self, data, cols, nbins=20, plot=True, plot_stddev=True, figsize=(7, 10)):
        """
        Generic Partial dependency plots for non-gbm models. 
        
        This will work with all model types
        """
        import h2o
        import pandas as pd
        import numpy as np
        h2o.h2o.no_progress()
        try:
            partial_df = []
            for col in cols:
                res_ls = []
                # determine bins for partial plot
                break_df = data[col].hist(breaks=20, plot=False).as_data_frame()
                break_df['breaks_shift'] = break_df['breaks'].shift()
                breaks_partial = list(break_df[['breaks_shift', 'breaks']].fillna(data[col].min()).to_records(index=False))
                breaks_partial[-1][1] = data[col].max()

                for idx, breaks in enumerate(breaks_partial):
                    lower = breaks[0]
                    upper = breaks[1]
                    lower_mask = data[col] >= lower
                    upper_mask = data[col] <= upper
                    res = self.model.predict(data[lower_mask and upper_mask])[:, -1]
                    actual = data[lower_mask and upper_mask, "response"].mean()[0]
                    mean_res = res.mean()[0]
                    std_res = res.sd()[0]
                    res_ls.append(((lower+upper)/2, mean_res, std_res, actual))
                # create dataframe so that we can plot
                res_df = pd.DataFrame(res_ls, columns=['idx', 'mean', 'std', 'actual'])
                res_df['std'] = np.maximum(self.model.predict(data)[:, -1].sd()[0], res_df['std'])
                res_df['lower'] = res_df['mean'] - res_df['std']
                res_df['upper'] = res_df['mean'] + res_df['std']
                partial_df.append(res_df.copy())
                if plot:
                    plt.figure(figsize=(7,10))
                    plt.plot(res_df['idx'], res_df['lower'], 'b--', 
                             res_df['idx'], res_df['upper'], 'b--', 
                             res_df['idx'], res_df['mean'], 'r-', 
                             res_df['idx'], res_df['actual'], 'go', )
                    plt.grid()
                    plt.show()
        except Exception as e:
            print(e)
        h2o.h2o.show_progress()
        return partial_df


# In[4]:


# problem setup

from sklearn.datasets import make_classification
X, y = make_classification(n_samples = 1000, n_features=10, n_informative=6, random_state=42)
X_df = pd.DataFrame(X)
X_df.columns = ["c{}".format(x) for x in list(X_df.columns)]

feature_names = list(X_df.columns)
class_labels = 'response'

train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X_df, y, train_size=0.80)

train_h2o_df = h2o.H2OFrame(train)
train_h2o_df.set_names(feature_names)
train_h2o_df['response'] = h2o.H2OFrame(y[labels_train])
train_h2o_df['response'] = train_h2o_df['response'].asfactor()

test_h2o_df = h2o.H2OFrame(test)
test_h2o_df.set_names(feature_names)
test_h2o_df['response'] = h2o.H2OFrame(y[labels_test])
test_h2o_df['response'] = test_h2o_df['response'].asfactor()


# In[5]:


mandelon_drf = H2OAutoML(max_runtime_secs = 30)


# In[6]:


mandelon_drf.train(x=feature_names,
         y='response',
         training_frame=train_h2o_df)


# In[7]:


# creating the explainer...
train_pandas_df = train_h2o_df[feature_names].as_data_frame() 
train_numpy_array = train_pandas_df.as_matrix() 

test_pandas_df = test_h2o_df[feature_names].as_data_frame() 
test_numpy_array = test_pandas_df.as_matrix() 

explainer = lime.lime_tabular.LimeTabularExplainer(train_numpy_array,
                                                   feature_names=feature_names,
                                                   class_names=['False', 'True'],
                                                   discretize_continuous=True)
h2o_drf_wrapper = h2o_predict_proba_wrapper(mandelon_drf, feature_names) 

i = 27
exp = explainer.explain_instance(test_numpy_array[i], h2o_drf_wrapper.predict_proba, num_features=2, top_labels=1)
exp.show_in_notebook(show_table=True, show_all=False)


# In[8]:


h2o_drf_wrapper.partial_dep_plot(train_h2o_df, [feature_names[5]])


# In[9]:


h2o_drf_wrapper.partial_dep_plot(train_h2o_df, [feature_names[5], feature_names[3]])


# In[10]:


#h2o.cluster().shutdown()

