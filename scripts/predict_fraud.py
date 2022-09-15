# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2022
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products 
#  made up of hundreds of individual components, each of which was 
#  individually copyrighted.  Each Cloudera open source product is a 
#  collective work under U.S. Copyright Law. Your license to use the 
#  collective work is as provided in your written agreement with  
#  Cloudera.  Used apart from the collective work, this file is 
#  licensed for your use pursuant to the open source license 
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute 
#  this code. If you do not have a written agreement with Cloudera nor 
#  with an authorized and properly licensed third party, you do not 
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED 
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO 
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND 
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU, 
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS 
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE 
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES 
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF 
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import cdsw
import numpy as np
import xgboost as xgb


# Inference here is performed without a Dask cluster so we use standard XGBoost calls

# Load the model trained in the distributed-xgboost-with-dask-on-cml notebook.
booster = xgb.Booster(model_file='/home/cdsw/model/best-xgboost-model')
threshold = 0.35

def predict_fraud(args):
    """
    args must contain the `features` key with a corresponding numpy array containing 
    the 29 features required for the XGBoost model. A sample set of features is shown 
    below and can be used for testing. 
    
    This XGBoost model outputs a continuous value between 0.0 and 1.0. A threshold
    must first be applied to this value to determine whether the sample should be 
    classified as fraud or not fraud.  
    
    The value of this threshold should be determined by the business use case. 
    Here, we provide a reasonable threshold. It would also be possible to add the 
    threshold to the args as a key:value pair. 
    """
    prediction = booster.inplace_predict(np.array(args['features']))
    if prediction[0] <= threshold: 
        return 0 # not fraud
    return 1 # fraud



# Offline testing - uncomment the following lines to test in a Session

sample_features = [[-1.35980713e+00, -7.27811733e-02,  2.53634674e+00,
                    1.37815522e+00, -3.38320770e-01,  4.62387778e-01,
                    2.39598554e-01,  9.86979013e-02,  3.63786970e-01,
                    9.07941720e-02, -5.51599533e-01, -6.17800856e-01,
                    -9.91389847e-01, -3.11169354e-01,  1.46817697e+00,
                    -4.70400525e-01,  2.07971242e-01,  2.57905802e-02,
                    4.03992960e-01,  2.51412098e-01, -1.83067779e-02,
                    2.77837576e-01, -1.10473910e-01,  6.69280749e-02,
                    1.28539358e-01, -1.89114844e-01,  1.33558377e-01,
                    -2.10530535e-02,  1.49620000e+02]]
#sample = {"features":sample_features}
#predict_fraud(sample)
