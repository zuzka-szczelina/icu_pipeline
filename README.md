## raw ICU data processing pipeline & ML usage trial

In this project I processed raw medical documentation obtained
in the form of paper cards.\
Documentation was obtained form children's hospital Intensive Care Unit.\
Work included:

1. Parameters medical meaning determination
2. Conversion of hand-written symbolic and numeric records to numerical dataset of time series
3. Dataset processing pipeline (visualisations, cleaning, interpolation, adaptation for ML usage)
4. Determination of how moments when a child's condition becomes unstable
   or deteriorate are reflected in the data (how anomalies look like)
5. Usage of obtaied time series as RNN training data (anomalies detection trial)

At all stages multiple consultations were needed with an experienced nurse
working in the Intensive Care Unit from which the data were obtained.


# /v2:
1st paragraph:

In this study raw medical data from the Cardiac Intensive Care Unit were processed. Data processing led to creation of a dataset adapted for the use in machine learning techniques. The obtained data were used to simulate the detection of anomalous conditions in a medical patient. In the experiment,  Recurrent Neural Networks of the LSTM type were used. Results were improved using fine-tuning. During the work, collaboration was with a Nurse working in the department from which the data was obtained. Work was done combining the knowledge of the nursing community and the knowledge of data processing and application. The result of the work is not a system capable of reliably predicting anomalous states in patients of the aforementioned ICU unit. It does, however, provide a cross-section of the subsequent processing of raw data and an example of the application of medical data to machine learning techniques. This work can serve as an example of an approach to using advanced data analysis techniques for medical purposes and provide a basis for further work with analogous data examples.



2nd parahraph:\
Following notebooks included:

#### 1. Creation of initial dataframe.
Paper cards containing medical records were translated into excel sheets. Translation involved multiple consultations with Nurse Expert. Aforementioned excel sheets were used to form initial dataframe containing all obtained records.
#### 2. Parameters visualisation.
Obtained parameters had a form of time series. All of them were visualised in the vorm of time plots.
#### 3. Parameters selection.
Taking into account the quality of data and Nurse Expert knowledge on their medical significance few parameters were selected for further project parts.
#### 4. Missing valuses exploration.
Conut and percentage of missing values in each time series were checked. Distribution of missing values in each series was presented in the form of time plot.
#### 5. Handling missing values part I - Linear Interpolation.
To use time series as training data for e.g. Recurrent Neural Network missing values presence needs to be handled somehow. One way of handling missing values is to perform data interpolation. First approach tried was linear interpolation. Results were visualised.
#### 6. Handling missing values part II - Interpolation Using Sesonal Decomposition.
Data at hand were time series describing vital signs of a patient observed for multiple days and recorded at on average hourly basis some periodical trends could be expected.

To handle that next approach was as follows: 
1. At first perform seasonal decomposition of each time series
2. Extract the established seasonal component
3. Perform linear interpolation on deseasonalised time series
4. Recreate time series adding extracted seasonal component.

Before the performace of seasonal decomposition constant time step was established for each time series.
#### 7. Correlation exploratoin.
Scaterrplot matrix was created to briefly explore correlation between parameters. 
#### 8. Approach to augmentation of time series.
It was not certain whether we would receive medical records of more patients. For this reason an attempt to data augmentation was made.\
Augmentation method that was used bases on altering the original series. Alteration was done by performing Fast Fourier Transform of the series, adding a random noise to frequency spectrum and performing Reverse Fast Fourier Transform to achieve modified series.
#### 9. Forecasting, without fine-tuning
The idea behind the work was to simulate the situation when:
1. First, for specified number of hours patients observation data are collected
2. Then the Neural Network (NN) is trained on collected data
3. For further time steps NN predicts an expected parameter value and the real value is collected simultaneously
4. If collected value is significantly different than the NN predicted one an alert is raised pointing an anomalous condition

Real location of the anomalies was pointed by the Nurse Expert, therefore achieved results of anomalies forecasting could be compared to their real placement.

Long short-term memory (LSTM) Recurrent Neural Networks were used.\
Experiment was run for various data sequence lengths used for model training and predictionand two different network models.
All results vere visualised.
#### 10. Forecasting, with fine-tuning
Lastly, fine-tuning was added.
After specified number of time steps model fine-tuning was performed to include additional collected observation data.
Visible difference in predictions appeared. For some parameters configuration model was able to predict all three true anomalies.


---
On the basis of the work included in this repository I wrote my Engineering Thesis. Thesis paper (in polish) is included [here](https://github.com/zuzka-szczelina/icu_pipeline/blob/master/engineering_thesis_paper/pl/projekt_inzynierski_Zuzanna_Szczelina.pdf)