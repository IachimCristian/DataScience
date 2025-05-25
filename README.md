# NYC Taxi Data Analysis Dashboard

An interactive web dashboard for visualizing machine learning models performed on NYC taxi data.

## Features

- Interactive model comparison with performance metrics
- Clustering visualization with adjustable parameters
- Regression analysis visualization
- Data explorer for custom scatter plots
- Ability to adjust model parameters

## Installation

1. Clone this repository
2. Install the required packages:

```
pip install -r requirements.txt
```

3. Make sure you have the NYC taxi dataset file (`nyc_taxi_final.csv`) in the project root directory.

### Optional: TensorFlow Support

If you want to use the Deep Learning model feature, uncomment the TensorFlow line in requirements.txt:

```
# tensorflow  # Uncomment if needed for deep learning models
```

Then install the updated requirements:

```
pip install -r requirements.txt
```

**Note for Windows users:** If you encounter DLL load errors with TensorFlow, you may need to:
- Install the Visual C++ Redistributable packages
- Use a compatible version of TensorFlow for your Python version
- See the [TensorFlow installation troubleshooting guide](https://www.tensorflow.org/install/errors)

The dashboard will work without TensorFlow, but the Deep Learning model option will be disabled.

## Running the Dashboard

Run the dashboard with:

```
python dashboard.py
```

Then open your web browser to `http://127.0.0.1:8050/` to view the dashboard.

## Dashboard Sections

1. **Model Performance**: Compare different machine learning models' accuracy
2. **Clustering**: Visualize KMeans and DBSCAN clustering on the taxi data
3. **Regression**: View regression model performance
4. **Data Explorer**: Create custom scatter plots to explore relationships between features

## Model Parameters

You can adjust various model parameters:
- KNN: Number of neighbors (k)
- Random Forest: Number of estimators
- Gradient Boosting: Learning rate
- Deep Learning: Epochs (if TensorFlow is available)
- KMeans: Number of clusters
- DBSCAN: Epsilon and minimum samples

## Project Structure

- `main.py`: Original model implementation
- `dashboard.py`: Dash web application
- `src/`: Directory containing model implementations
- `requirements.txt`: Required Python packages 