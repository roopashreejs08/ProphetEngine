\# Shared Project Contract

\## Predictive Maintenance for Industrial Engines



\*\*Version:\*\* 1.0  

\*\*Random seed:\*\* 42  

\*\*Dataset:\*\* NASA C-MAPSS FD001 and FD003; AI4I 2020



\### 1. Remaining Useful Life (RUL) Definition



For each engine, calculate:



`RUL = maximum cycle of that engine - current cycle`



Clip the RUL label to a maximum of 125 cycles:



`RUL = min(maximum cycle - current cycle, 125)`



Apply this definition consistently to the NASA training data.



\### 2. Data Splitting



\- Split training and validation data by engine ID, never by individual rows.

\- Use random seed `42` for reproducibility.

\- Keep engines in the validation set separate from engines in the training set.

\- Do not use the test set for training or hyperparameter tuning.



\### 3. Feature Scaling



\- Fit the scaler using training data only.

\- Use the fitted scaler to transform validation and test data.

\- Never fit a scaler on validation or test data.



\### 4. Time-Series Windowing



\- Use a window length of 30 consecutive engine cycles.

\- Each window contains sensor and operating-condition features in chronological order.

\- Use the window's final cycle as the prediction point.

\- Keep windows within the same engine; never combine rows from different engines.



\### 5. Evaluation Metrics



\#### RUL Prediction



Every RUL model must report:

\- Root Mean Squared Error (RMSE)

\- NASA scoring function



\#### Failure Classification



Every classification model must report:

\- Precision

\- Recall

\- F1-score

\- Precision-Recall Area Under the Curve (PR-AUC)



Accuracy may be included as an additional metric but must not be the only classification metric.



\### 6. Model Interface



Every RUL model must expose the following logical interface:



`predict\_rul(last\_30\_rows) -> float`



Input: The most recent 30 consecutive cycles for one engine, with the expected feature columns and preprocessing applied consistently.



Output: One predicted remaining useful life value, expressed in cycles.



Each model must use the same agreed input-column order and preprocessing contract.



\### 7. Model Storage



\- Save trained model artifacts under `/models/`.

\- Save any required fitted scaler or preprocessing artifact.

\- Document the model name, dataset, features, and expected input shape.

\- Do not commit large raw datasets or model artifacts to Git unless the team explicitly agrees.



\### 8. Shared Results File



Maintain `/shared/results.csv` with these columns:



`model,dataset,RMSE,NASA\_score,seed`



Use one row per RUL model evaluation. Record the seed as `42` when applicable.



Classification results must also be recorded in a separate classification results file or additional agreed columns containing precision, recall, F1-score, and PR-AUC.



\### 9. Reproducibility and Leakage Prevention



\- Keep the random seed fixed at 42 for the agreed baseline experiments.

\- Keep validation and test data separate from model fitting.

\- Apply resampling methods, feature selection, and hyperparameter tuning only within the training process.

\- Ensure every reported metric comes from actual model evaluation.



\### 10. Team Agreement



All team members must follow this contract when preparing data, training models, evaluating predictions, and integrating the application.



Changes require team agreement and should be made only when a genuine technical issue is identified.

