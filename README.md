# GIS and Machine Learning Based Framework for Flood Risk Prediction and Dynamic Evacuation Routing in Chennai

## Overview

This project presents an integrated Geographic Information System (GIS) and Machine Learning (ML) framework for flood risk prediction and dynamic evacuation routing in Chennai.

The framework combines temporal flood/runoff prediction using Machine Learning, terrain-based flood susceptibility analysis using GIS, and graph-based route optimization to identify safer evacuation routes during flood events.

The project was developed as part of an IEEE conference research work.

---

## Objectives

The main objectives of the project are:

- Predict flood-related runoff using Machine Learning.
- Capture temporal patterns in rainfall, soil moisture, and runoff data.
- Identify flood-prone regions using GIS and terrain analysis.
- Perform zone-wise flood risk assessment.
- Integrate Machine Learning predictions with GIS-based flood risk information.
- Generate flood-aware evacuation routes.
- Compare XGBoost and LSTM approaches for runoff prediction.
- Support disaster management and evacuation planning.

---

# Technologies Used

## Programming

- Python 3.10+

## GIS

- QGIS
- Google Earth Engine
- WhiteboxTools

## Machine Learning

- XGBoost
- LSTM
- TensorFlow / Keras
- Scikit-learn

## Python Libraries

- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- tensorflow
- geopandas
- shapely
- pyproj
- networkx
- requests

---

# Project Workflow

```text
Rainfall Data
      │
      ▼
Soil Moisture Data
      │
      ▼
Runoff Data
      │
      ▼
Data Preprocessing
      │
      ▼
Feature Engineering
      │
      ▼
Lag Feature Creation
      │
      ├──────────────────────┐
      ▼                      ▼
  XGBoost                  LSTM
      │                      │
      ▼                      ▼
Runoff Prediction       Temporal Runoff
                         Prediction
      │                      │
      └──────────┬───────────┘
                 │
                 ▼
       Model Evaluation
                 │
                 ▼
       Flood Classification
                 │
                 ▼
       GIS Flood Risk Analysis
                 │
                 ▼
       Flood Risk + Road Network
                 │
                 ▼
        Flood-aware Cost Function
                 │
                 ▼
        Dijkstra's Algorithm
                 │
                 ▼
       Safe Evacuation Route