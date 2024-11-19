Brain MRI Segmentation Project
This project performs brain tumor segmentation using deep learning models such as UNet and DeepLabV3+. The goal is to identify and segment brain tumors in MRI images.

Dataset
The dataset used for this project is the Lower-Grade Glioma (LGG) Segmentation Dataset, which can be downloaded from Kaggle:
Kaggle Dataset Link : https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

How to Run
Step 1: Download the Dataset
Visit the Kaggle link provided above.
Download the dataset to your local machine.
Step 2: Run the Jupyter Notebook
Open the Main.ipynb file in Jupyter Notebook or any compatible environment.
Ensure that the dataset is properly placed, and the paths in the notebook are correctly configured.
Execute the notebook cells to train and evaluate the models.
Step 3: Run the Streamlit App
Navigate to the folder where app.py is located.
Open a terminal or Command Prompt.
Change the directory to the app folder by running:
cd path/to/your/app.py
Start the Streamlit app by typing:
streamlit run app.py
Open the Streamlit app in your browser to test brain tumor segmentation on MRI images.
