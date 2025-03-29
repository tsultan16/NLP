---------------------------------------------
* IMPORTANT GUIDE FOR RUNNING THE NOTEBOOKS *
---------------------------------------------

This folder contains 6 Jupyter notebooks and 4 python scripts in total. 

The 4 python scripts contain functions for data loading from the json files and also for cleaning/preprocessing the data. They also contain classes with our implementations of the custom BERT model, the WordPiece tokenizer and the BM25 model.

Each notebook follows the prescribed .ipynb template given in the assignment specs. 

Because of interdependencies between notebooks, they cannot be run in any arbitrary order. For example, to run the "Mon5PM_Group2_COMP90042_Project_2024_BERT_Multitask_Pretraining.ipynb" notebook, you first need to run the "Mon5PM_Group2_COMP90042_Project_2024_WordPiece_Tokenizer_Training.ipynb" notebook which will train the WordPiece tokenizer and save a copy of the tokenizer object to a pickle file contained in the same directory as the notebook. Then you can run the "Mon5PM_Group2_COMP90042_Project_2024_BERT_Multitask_Pretraining.ipynb" notebook from the same directory which will load the saved tokenizer object from file and make use of it for the multitask pre-training after which it will save a checkpoint of the pre-trained BERT model to file. 

Similarly, the notebook "Mon5PM_Group2_COMP90042_Project_2024_monoBERT.ipynb" should not be run until after the pretrained BERT model checkpoint has been saved and also the "Mon5PM_Group2_COMP90042_Project_2024_BM25.ipynb" notebook has been run as well. This is because the monoBERT retriever makes use of both the pre-trained BERT and the BM25 models. Once the monoBERT is trained and the evaluation is run, the retrieved evidence ids are also saved to file and later used by the classifier. So before running the "Mon5PM_Group2_COMP90042_Project_2024_Classifier_*.ipynb" notebooks, the monoBERT notebook has to be run and evaluation completed.

In summary, the notebooks should be run in the following order:

1) Mon5PM_Group2_COMP90042_Project_2024_WordPiece_Tokenizer_Training.ipynb
2) Mon5PM_Group2_COMP90042_Project_2024_BM25.ipynb
3) Mon5PM_Group2_COMP90042_Project_2024_BERT_Multitask_Pretraining.ipynb
4) Mon5PM_Group2_COMP90042_Project_2024_monoBERT.ipynb
5) Mon5PM_Group2_COMP90042_Project_2024_Classifier_Baseline.ipynb
6) Mon5PM_Group2_COMP90042_Project_2024_Classifier_Main.ipynb


***IMPORTANT***: ALL NOTEBOOKS MUST BE RUN FROM THE SAME RUNTIME DIRECTORY CONTAINING THE PYTHON SCRIPTS AND SAVED PICKLE FILES. ALSO, ANY EXTRA PACKAGES REQUIRED BY EACH NOTEBOOK IS AUTOMATICALLY PIP INSTALLED IN THE FIRST CODE CELL. 

The scripts are expected to be in the same directory as the notebooks, and all other files generated from running the notebooks also reside in the same directory. Also note that the dataset json files containing the training data are expected to be in a sub-directory called 'data/'. 

Finally, when training the BERT model, the dataset preparation involves some random cropping of sentences to make them fit within the 128 block size of the model. This may lead to small variations in the final results upon running the code at different times as we have not explicitly specified seeds for any of the random number generation.
