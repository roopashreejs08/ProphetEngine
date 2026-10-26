Dataset and Exploratory Data Analysis



1\. Dataset Overview



This project uses two datasets for predictive maintenance analysis:
NASA C-MAPSS for remaining useful life (RUL) analysis and AI4I 2020 for machine-failure classification. The datasets represent different predictive-maintenance tasks and are processed separately.



1.1 NASA C-MAPSS
The NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset is used to investigate engine degradation and estimate remaining useful life. The current workflow uses the FD001 and FD003 subsets.

Each training record represents an engine at a particular operating cycle. The data includes an engine identifier (engine\_id), cycle number (cycle), three operational settings (op\_setting\_1 to op\_setting\_3), and 21 sensor measurements (sensor\_1 to sensor\_21).The preprocessing workflow derives RUL labels from the training sequences and caps the labels at 125 cycles to limit the influence of very large RUL values.

The preprocessing pipeline identifies low-variance sensors and removes them from the windowed model inputs. For the current runs, six sensors were removed from FD001 (sensor\_1, sensor\_5, sensor\_10,sensor\_16, sensor\_18, and sensor\_19), and five were removed from FD003 (sensor\_1, sensor\_5, sensor\_16, sensor\_18, and sensor\_19). The remaining measurements are organized into 30-cycle windows for sequence-based modelling.

The training data is split by engine into training and validation groups, using a fixed random state of 42 and an 80/20 engine-level split. Splitting by engine helps keep windows from the same engine from appearing in both groups. In the current preprocessing runs, the generated arrays have shapes of (14241, 30, 18) for FD001 training and (3490, 30, 18) for FD001 validation; for FD003, the shapes are (17692, 30, 19) for training and (4128, 30, 19) for validation.

The RUL distribution plots show the effect of the 125-cycle cap: values at the cap form a visible concentration in the distribution. This concentration is a result of the label-capping procedure and should not be interpreted as evidence that engines naturally cluster at exactly 125 cycles remaining.



1.2 AI4I 2020 Predictive Maintenance Dataset
The AI4I 2020 dataset is used for machine-failure classification. The raw file contains 10,000 records and 14 columns, including identifiers,machine type, operating measurements, the overall Machine failure target, and five failure-mode indicators: TWF, HDF, PWF, OSF,and RNF.

The preprocessing pipeline removes UDI and Product ID from model inputs and encodes the categorical Type field using one-hot encoding.The resulting model input contains eight features: five numerical measurements---air temperature, process temperature, rotational speed,torque, and tool wear---and the encoded type indicators (Type\_H,Type\_L, and Type\_M).

The data is split into training and validation sets using an 80/20 stratified split with a fixed random state of 42. Stratification preserves approximately the same target proportions in both subsets. The training subset contains 8,000 records, including 271 failures; the validation subset contains 2,000 records, including 68 failures.

The failure-mode columns are retained in the cleaned data for descriptive analysis but excluded from the model input features. This is important because those columns describe specific failure outcomes and could leak target-related information into a predictive model. The overall Machine failure column is the target, not an input feature.



2\. Exploratory Data Analysis



2.1 AI4I Class Imbalance
The class-imbalance chart shows 9,661 observations without machine failure and 339 observations with machine failure. Failures therefore represent approximately 3.39% of the dataset. This is a strongly imbalanced target, so model evaluation should not rely on accuracy alone. Suitable classification metrics should also reflect failure detection, such as precision, recall, F1-score, and a confusion matrix.Any resampling or class-weighting strategy should be applied using training data only.



2.2 AI4I Correlation Heatmap
The correlation heatmap summarizes pairwise linear relationships between the numerical measurements, encoded variables, target, and the columns included in the EDA plot. It provides a descriptive view of relationships in the dataset, but correlation does not establish causation.

The heatmap may include failure-mode indicators for exploratory analysis. Their presence in a descriptive plot does not mean they are permitted as model inputs. The preprocessing pipeline excludes TWF,HDF, PWF, OSF, and RNF from the AI4I training and validation features.



2.3 C-MAPSS RUL Distributions
Separate RUL distribution figures are produced for FD001 and FD003. Both distributions reflect the label-generation procedure, including the capof 125 cycles. The concentration at the cap is expected under this preprocessing rule. The plots help describe the target values supplied to the sequence models, but they do not independently demonstrate model performance.



2.4 C-MAPSS Sensor Trends
The FD001 and FD003 sensor-trend figures visualize selected sensor measurements over engine cycles for example engines. Some plotted sensors show upward or changing patterns over the displayed cycles,while others remain relatively stable. These plots are useful for inspecting sensor behaviour and identifying candidate degradation signals.

The plots should be interpreted as examples rather than proof that the same trend occurs in every engine. Sensor readings can vary by engine and operating conditions, and not every sensor is expected to provide a clear monotonic degradation signal. Broader conclusions would require analysis across more engines or aggregated trends.



3\. Summary and Limitations

The exploratory analysis establishes the main properties relevant to the project: C-MAPSS provides cycle-based multivariate sensor sequences for RUL modelling, while AI4I provides labelled records for machine-failure classification. The AI4I target is imbalanced, the C-MAPSS RUL labels are capped at 125 cycles, and selected C-MAPSS sensors exhibit different behaviours across the example plots.

The current work covers FD001 and FD003 for C-MAPSS. Handling FD002 and FD004 with scaling conditioned on operating regimes remains an optional stretch goal and is not included in the completed core workflow. The figures are exploratory and do not, by themselves, establish predictive performance. Final model comparisons should be reported separately using held-out evaluation data and appropriate task-specific metrics.

