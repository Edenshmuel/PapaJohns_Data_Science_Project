# Papa John's Data Science Project
Welcome to the Papa John’s Data Science Project!

This repository is a journey into understanding sales patterns and making data-driven decisions for Papa John’s. Through thoughtful analysis and practical modeling, we’ve explored how different categories and sales channels contribute to overall performance.

**Please note:**
The data used in this project is not included in this repository. Due to confidentiality and security reasons, and at the request of Papa John’s, we are unable to publish the original datasets.

# Our Approach
To forecast demand for each product category, we designed a complete pipeline—from data preparation to dashboard deployment. As part of the process, we built tools to automatically extract product information from text descriptions, ensuring reliable categorization and cleaner analysis.

Our feature engineering focused on what really matters:

- Time-based features: Is this a Jewish or Christian holiday? Is it a weekend? We even used weekly and monthly sinusoidal/cosine transformations to capture seasonality.
- Product features: Category membership and other product-specific details were included to enrich our models.
  
# Modeling
We compared several advanced forecasting models, including:

- TFT (Temporal Fusion Transformer)
- DeepAR
- Prophet
- XGBoost
- XGBoost + SARIMA
  
After thorough testing, DeepAR proved to be the best model for our data and goals.

# From Data to Decisions
At the heart of this project is a dashboard designed for branch managers. With accurate sales forecasts at their fingertips, managers can make smarter decisions—whether it’s ordering the right amount of raw materials or optimizing staff schedules for busy days.

We invite you to explore our notebooks and approach. Whether you’re a data enthusiast, a business professional, or just curious, you’ll find a clear story about how data can shape better decisions.
