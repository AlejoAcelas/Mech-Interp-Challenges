# Mechanistic Interpretability Challenges

This repository contains the starting materials for the Capture the Flag Mechanistic Interpretability Challenges. For more information on the challenges and their purpose you can visit the [announcement post](https://www.lesswrong.com/posts/SbLRsdajhMQdAbaqL/capture-the-flag-mechanistic-interpretability-challenges) on LessWrong.  

Here's a short description of the files and folders in the repo:  
* **demo.py:** Loads the models corresponding to the three challenges and evaluates them on a sample dataset
* **dataset_public.py:** Defines some dataset classes matching the task of each model. They're different from the datasets used for training the models. 
* **scoring.py:** A slighly modified copy of the file used to score submissions
* **/models:** Contains the weights of the models used for each challenge.
* **/submission_example:** an example submission for the three challenges. Contains a simple baseline for each challenge.  
* **model.py:** Defines custom function to instantiate a transformer model using TransformerLens.  

You can submit your challenge solutions at the following [CodaBench Competition](https://www.codabench.org/competitions/1275/). 
