To run the program, NASrun.py is executed. Currently cnnnas.py does a manual auto-recursive prediction to get 30 minute predictions, sliding throughout the whole test set. However, it is actually training each child network epoch on 5 minute predictions, so it provides a training MAE based on 5 minute performance, and a 30 minute prediction performance "Prediction MAE: " based on the autorecursive loops. 

Remember, currently the patient is on 552, which has a standard deviation of 54.58, and the MAE values being printed are standardized.
