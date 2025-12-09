## Description

In this study, I processed raw medical data from the Cardiac Intensive Care Unit. 
Data processing led to creation of a dataset adapted for use in machine learning techniques. 
I used the obtained data to simulate the detection of anomalous conditions in a medical patient. 

In the experiment, I used Recurrent Neural Networks of the LSTM type. 
I was able to improve the results using fine-tuning. 
In the course of the work, I collaborated with a Nurse working in the department from which the data was obtained. 
Work was done combining the knowledge of the nursing community and the knowledge of data processing and application.

The result of the work is not a system capable of reliably
predicting anomalous states in patients of the aforementioned ICU unit. 
It does, however, provide a cross-section of the subsequent processing of raw data
and an example of the application of medical data to machine learning techniques. 
This work can serve as an example of an approach to using advanced data analysis techniques
for medical purposes and provide a basis for further work with analogous data examples.


## How to read

- Toget familiar with results, theoretical introduction and description in the form of article one should read the [paper](https://github.com/zuzka-szczelina/icu_pipeline/tree/master/engineering_thesis_paper),
which is my engineering thesis paper.
- To get familiar with the code work one should read the content of [notebooks](https://github.com/zuzka-szczelina/icu_pipeline/tree/master/source_code/notebooks) one after another. 
Notebooks are numbered and explanations of successive steps is provided in each notebook.


## Paper content

1. Introduction
1.1 Origin of the Data \
1.2 Description of Selected Medical Parameters\
1.3 Formal Requirements When Working with Medical Data\
1.4 Medical Consultation \
1.5 Recurrent Neural Networks (RNN) and LSTM Networks

2. Data Processing\
2.1 Extraction of Information from Records\
2.2 Creation of the Initial Dataset\
2.3 Visualization\
2.4 Selection of Parameters\
2.5 Missing Values\
2.6 Linear Interpolation\
2.7 Interpolation Using Seasonal Decomposition\
2.8 Correlations\
2.9 Data Augmentation

3. Anomalous conditions Prediction\
3.1 Forecasting Without Fine-Tuning\
3.2 Forecasting With Fine-Tuning

4. Comparison of Results\
4.1 Real Data\
4.1.1 Fine-Tuning vs. No Fine-Tuning\
4.1.2 Model Comparison\
4.1.3 Comparison of Sequence Lengths\
4.2 Optimization\
4.3 Generated Data

5. Conclusion\
5.1 Summary\
5.2 Main Tools\
5.3 Medical Consultation

6. Bibliography


## Notebooks content

1. Initial dataset creation
2. Parameters visualisation
3. Selection of parameters
4. Missing values exploration
5. Linear interpolation of data
6. Interpolation using seasonal decomposition
7. Correlation visualisation
8. Data augmentaion approach
9. Anomalous conditions prediction: forecasting without finetuning
10. Anomalous conditions prediction: forecasting using fine tuning
