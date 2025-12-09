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

- To get familiar with results, theoretical introduction and description in the form of article one should read the [paper](https://github.com/zuzka-szczelina/icu_pipeline/tree/master/engineering_thesis_paper),
which is my engineering thesis paper  (written in polish).
- To get familiar with the code work one should read the content of [notebooks](https://github.com/zuzka-szczelina/icu_pipeline/tree/master/source_code/notebooks) one after another. 
Notebooks are numbered and explanations of successive steps is provided in each notebook.


## Paper content

1. Introduction\
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

1. Initial dataset creation.\
Paper cards containing medical records were translated into excel sheets. Translation involved multiple consultations with Nurse Expert. Aforementioned excel sheets were used to form initial dataframe containing all obtained records.
2. Parameters visualisation.\
Obtained parameters had a form of time series. All of them were visualised in the vorm of time plots.
3. Selection of parameters.\
Taking into account the quality of data and Nurse Expert knowledge on their medical significance few parameters were selected for further project parts.
4. Missing values exploration.\
Conut and percentage of missing values in each time series were checked. Distribution of missing values in each series was presented in the form of time plot.
5. Linear interpolation of data.\
To use time series as training data for e.g. Recurrent Neural Network missing values presence needs to be handled somehow. One way of handling missing values is to perform data interpolation. First approach tried was linear interpolation. Results were visualised.
6. Interpolation using seasonal decomposition.\
Data at hand were time series describing vital signs of a patient observed for multiple days and recorded at on average hourly basis some periodical trends could be expected.\
To handle that next approach was as follows: 
- I. At first perform seasonal decomposition of each time series
- II. Extract the established seasonal component
- III. Perform linear interpolation on deseasonalised time series
- IV. Recreate time series adding extracted seasonal component.\
Before the performace of seasonal decomposition constant time step was established for each time series.
7. Correlation visualisation.\
Scaterrplot matrix was created to briefly explore correlation between parameters. 
8. Data augmentaion approach.\
It was not certain whether we would receive medical records of more patients. For this reason an attempt to data augmentation was made.\
Augmentation method that was used bases on altering the original series. Alteration was done by performing Fast Fourier Transform of the series, adding a random noise to frequency spectrum and performing Reverse Fast Fourier Transform to achieve modified series.
9. Anomalous conditions prediction: forecasting without finetuning.\
The idea behind the work was to simulate the situation when:
- I. First, for specified number of hours patients observation data are collected
- II. Then the Neural Network (NN) is trained on collected data
- III. For further time steps NN predicts an expected parameter value and the real value is collected simultaneously
- IV. If collected value is significantly different than the NN predicted one an alert is raised pointing an anomalous condition.\
Real location of the anomalies was pointed by the Nurse Expert, therefore achieved results of anomalies forecasting could be compared to their real placement.\
Long short-term memory (LSTM) Recurrent Neural Networks were used.\
Experiment was run for various data sequence lengths used for model training and predictionand two different network models.
All results vere visualised.
10. Anomalous conditions prediction: forecasting using fine-tuning.\
Lastly, fine-tuning was added.
After specified number of time steps model fine-tuning was performed to include additional collected observation data.
Visible difference in predictions appeared. For some parameters configuration model was able to predict all three true anomalies.
