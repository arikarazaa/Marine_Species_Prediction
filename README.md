# Marine_Species_Prediction
Marine Species Population Prediction uses machine learning (Linear Regression &amp; Random Forest) to model and forecast global marine species populations from 1950–2023 using FAO fisheries data. It achieves up to 99% predictive accuracy and generates short-term catch-based forecasts through 2030 across 2,700+ species worldwide.


Marine Species Population Prediction
Overview
This project uses machine learning to model and predict global marine species populations based on historical fisheries data (1950–2023).
Fish catch volumes are used as a proxy for population trends, enabling short-term forecasting under the assumption of constant fishing pressure. The project explores whether fisheries exhibit predictable dynamics and what factors drive population changes across species.
________________________________________
 Objectives
•	Model historical population trends of marine species using fisheries data 
•	Evaluate the effectiveness of machine learning in ecological forecasting 
•	Identify key drivers of population dynamics 
•	Generate short-term forecasts (2024–2030) 
________________________________________
 Dataset
•	Source: FAO FishStatJ – Global Capture Production 
•	Time Period: 1950–2023 
•	Raw Observations: ~1,000,000+ 
•	Cleaned Dataset: ~86,000 records 
•	Coverage: 2,700+ marine species across global regions 
________________________________________
 Methodology
Data Processing
•	Integrated multiple FAO datasets 
•	Removed inconsistent and missing observations 
•	Aggregated data into species × region × year format 
Feature Engineering
•	Lag features (1-year, 2-year) 
•	Rolling averages (5-year trends) 
•	Year-on-year percentage changes 
•	Encoding of biological and regional variables 
•	Log transformation to stabilize variance 
Model Training
•	Time-based split: 
o	Training: Pre-2016 
o	Testing: 2016–2023 
•	Models used: 
o	Linear Regression (baseline) 
o	Random Forest Regression 
________________________________________
 Results
Model Performance
Model	R² Score	Insight
Linear Regression	~0.82	Captures general trends but misses non-linear behavior
Random Forest	~0.99	Captures complex and non-linear population dynamics
⚠️ Note: The very high R² suggests strong short-term predictability but may also reflect temporal dependency in the data.
________________________________________
Key Insights
•	Marine populations behave as momentum-driven systems 
•	Past catch values are the strongest predictors of future trends 
•	Non-linear models significantly outperform linear approaches 
•	Biological characteristics matter more than geographic location 
•	Short-term forecasting is feasible, but long-term predictions require environmental and policy variables 
________________________________________
Case Studies
•	Atlantic Cod → Long-term decline 
•	Anchoveta → High variability (climate-driven, e.g., El Niño) 
•	Skipjack Tuna → Stable increasing trend 
These examples highlight that species dynamics are highly heterogeneous.
________________________________________
Forecasting (2024–2030)
•	Iterative predictions using trained models 
•	Key observations: 
o	Managed species show relative stability 
o	Climate-sensitive species exhibit volatility 
o	No evidence of large-scale recovery under current conditions 
________________________________________
Visualizations
•	Predicted vs actual scatter plots 
•	Model comparison charts 
•	Feature importance graphs 
•	Species-specific trend analysis 
•	Forecast projections 
________________________________________ Tech Stack
•	Python 
•	Pandas, NumPy 
•	Scikit-learn 
•	Matplotlib, Plotly 
________________________________________ Installation
git clone https://github.com/your-username/marine-ml-prediction.git
cd marine-ml-prediction
pip install pandas numpy scikit-learn matplotlib plotly
________________________________________
Usage
jupyter notebook
Run the notebook to reproduce:
•	Data preprocessing 
•	Model training 
•	Forecasting 
________________________________________
Future Work
•	Incorporate fishing effort data (to separate effort vs population decline) 
•	Include climate variables (SST, ENSO index) 
•	Develop species-specific models 
•	Apply framework to regional ecosystems (e.g., Red Sea fisheries) 
•	Integrate geospatial analysis 
________________________________________
Author
Arika Raza
Dataset provided by the Food and Agriculture Organization (FAO).

