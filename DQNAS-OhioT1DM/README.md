Double Q-Network Neural Architecture Search for OhioT1DM Dataset
Notes:
- This repository was heavily adapted from DQNAS: Neural Architecture Search using
Reinforcement Learning by Anshumaan Chauhan at https://github.com/Anshumaan-Chauhan02/DQNAS
- Models are saved as "5___weights.keras" files, and opened in "usage.py" or "usage_render_research_images.py"

File Summaries:
CNNCONSTANTS.py - This file holds constants for training parameters like controller epoch numbers, child-network epoch limit, child-network architecture length, etc.
CNNGenerator.py - This file is mainly used for setting up the child neural networks from a defined search-space vocabulary. It compiles the models, and also trains them when called.
cnnnas.py - This file acts as the main processor for doing NAS. It calls functions to train the child networks, as well as the controllers (main Q-network and target Q-network). It also keeps track of child-network training results (RMSE and MAE) and stores it/uses it to call training functions on controller networks.
DQNController.py - This file creates and compiles the controller Q-network models. It is also responsible for sampling all child models.
manual_nn_trainer.py - After superior child models were found with NAS and simple learning rate/batch size/alpha parameters, this file was used to fine-tune the models with a grid search of different values in those parameters.
manual_nn_trainer_saver.py - This file does the same as the former, but was specifically used to save the files/weights for this repository.
NASrun.py - This file establishes the search-space and controllers, and then searches, based on CNNCONSTANTS and all other imported files. It is the top-level run file.
NASutils.py - A few helper functions reside in this file.
time_series2.py - This file establishes the dataset for each patient in a time-series format for the sake of batching and windowing. It was adapted from TensorFlow's time-series tutorial.
usage.py - This is the main file to run to demonstrate usage of the state-of-the-art models. It loads each patient's personalized model consecutively and has it predict on the test set for that patient. It plots the results, and prints error values.
usage_render_research_images.py - On top of the former file, I included the more specific usage of plotting/rendering that was used for this accompanying paper.
xml_reader.py - This file converts the Ohio University OhioT1DM datasets into a list, which time_series2.py later converts to a dataframe for ML usage.
