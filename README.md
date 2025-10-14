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
