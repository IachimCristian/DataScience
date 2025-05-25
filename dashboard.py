import dash
from dash import dcc, html, Input, Output, callback, State, MATCH, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import importlib.util

# Import the model functions
from src.knn_custom import KNNFast
from src.supervised_models import train_logistic_regression
from src.ensemble_models import train_random_forest, train_gradient_boosting
from src.clustering import run_kmeans, run_dbscan
from src.regression import train_random_forest_regressor, evaluate_regression
from src.cross_validation import perform_2fold_cv, compare_models_2fold_cv

# Check if TensorFlow is available
tensorflow_available = importlib.util.find_spec("tensorflow") is not None
try:
    if tensorflow_available:
        from src.deep_learning import build_deep_learning_model
except ImportError:
    tensorflow_available = False
    print("TensorFlow could not be imported. Deep Learning model will be disabled.")

# Create static folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

# Initialize the Dash app
app = dash.Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap',
        'https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap'
    ],
    assets_folder='static',
    suppress_callback_exceptions=True  # This is important for dynamic components
)
server = app.server
app.title = "NYC TAXI DATA ANALYTICS"

# Theme colors for consistent plotting
colors = {
    'background': '#222834',
    'text': '#e0e0e0',
    'primary': '#e5b83b',
    'secondary': '#3498db',
    'accent': '#e74c3c',
    'grid': '#333d51'
}

# Load and prepare data
def load_data():
    df = pd.read_csv("data/nyc_taxi_final.csv")
    return df

def get_features_and_targets():
    df = load_data()
    feature_cols = ['trip_distance', 'fare_amount', 'total_amount', 'tolls_amount',
                  'pickup_hour', 'pickup_day', 'pickup_weekday', 'pickup_month',
                  'trip_duration', 'speed_mph', 'is_weekend', 'is_rush_hour', 'is_night',
                  'pulocationid', 'passenger_count', 'payment_type', 'improvement_surcharge',
                  'tip_amount', 'mta_tax', 'extra']
    
    X = df[feature_cols].values
    y_class = df['high_fare'].values
    y_reg = df['fare_amount'].values
    
    return X, y_class, y_reg, feature_cols, df

# Define model options based on available packages
model_options = [
    {'label': 'KNN', 'value': 'knn'},
    {'label': 'Logistic Regression', 'value': 'logistic'},
    {'label': 'Random Forest', 'value': 'random_forest'},
    {'label': 'Gradient Boosting', 'value': 'gradient_boosting'},
]

# Add Deep Learning option only if TensorFlow is available
if tensorflow_available:
    model_options.append({'label': 'Deep Learning', 'value': 'deep_learning'})

# Create a gauge chart
def create_gauge_chart(value, title="", min_val=0, max_val=100):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'color': colors['text']}},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': colors['text']},
            'bar': {'color': colors['primary']},
            'bgcolor': colors['background'],
            'borderwidth': 2,
            'bordercolor': colors['text'],
            'steps': [
                {'range': [min_val, max_val * 0.7], 'color': colors['background']},
                {'range': [max_val * 0.7, max_val * 0.9], 'color': '#3498db'},
                {'range': [max_val * 0.9, max_val], 'color': '#e74c3c'}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor=colors['background'],
        font={'color': colors['text'], 'family': "Roboto"},
        margin=dict(l=10, r=10, t=30, b=10),
        height=200
    )
    
    return fig

# Create a pie chart for classification results
def create_pie_chart(values, labels, title=""):
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=[colors['primary'], colors['secondary'], colors['accent']]
    ))
    
    fig.update_layout(
        title=title,
        paper_bgcolor=colors['background'],
        plot_bgcolor=colors['background'],
        font={'color': colors['text'], 'family': "Roboto"},
        legend=dict(
            font=dict(color=colors['text']),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Get some data for initial KPIs
try:
    _, _, _, _, df = get_features_and_targets()
    avg_distance = np.mean(df['trip_distance'])
    avg_fare = np.mean(df['fare_amount'])
    max_fare = np.max(df['fare_amount'])
    high_fare_pct = np.mean(df['high_fare']) * 100
except:
    avg_distance = 0
    avg_fare = 0
    max_fare = 0
    high_fare_pct = 0

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H3("NYC TAXI DATA ANALYTICS DASHBOARD", className="mt-3"),
            html.P("Machine Learning Models and Data Exploration", className="text-muted")
        ], width=9),
        dbc.Col([
            html.Div([
                html.A("LEARN MORE", href="#", className="btn btn-sm btn-outline-light"),
                html.Img(src="https://plotly.com/dash/", height="20px", className="ml-2")
            ], className="d-flex justify-content-end align-items-center h-100")
        ], width=3)
    ], className="mb-4 border-bottom border-dark pb-2"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("SPECIFICATION SETTINGS"),
                dbc.CardBody([
                    html.Div([
                        html.Div("Model Parameters", className="mb-2 font-weight-bold"),
                        dcc.Dropdown(
                            id='model-dropdown',
                            options=model_options,
                            value='knn',
                            className="mb-3"
                        ),
                        html.Div(id='model-params-container', className='mb-3'),
                        html.Button('RUN MODEL', id='run-model-button', className='btn btn-primary mb-4 w-100')
                    ]),
                    
                    html.Div([
                        html.Div("Clustering Method", className="mb-2 font-weight-bold"),
                        dcc.Dropdown(
                            id='clustering-dropdown',
                            options=[
                                {'label': 'KMeans', 'value': 'kmeans'},
                                {'label': 'DBSCAN', 'value': 'dbscan'}
                            ],
                            value='kmeans',
                            className="mb-3"
                        ),
                        html.Div(id='clustering-params-container', className='mb-3'),
                        html.Button('RUN CLUSTERING', id='run-clustering-button', className='btn btn-primary w-100')
                    ])
                ])
            ], className='mb-4'),
            
            dbc.Card([
                dbc.CardHeader("PERFORMANCE METRICS"),
                dbc.CardBody([
                    html.Div([
                        html.Div("Operator ID", className="kpi-label"),
                        html.Div("T104", className="kpi-value mb-4")
                    ], className="kpi-container"),
                    
                    html.Div([
                        html.Div("Time to completion", className="kpi-label mb-2"),
                        dcc.Graph(
                            id='gauge-chart',
                            figure=create_gauge_chart(50.0, "", 0, 100),
                            config={'displayModeBar': False}
                        )
                    ], className="gauge-container")
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("CONTROL CHARTS DASHBOARD"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Div("Parameter", className="text-center font-weight-bold"),
                                html.Div("Count", className="text-center font-weight-bold"),
                                html.Div("Baseline", className="text-center font-weight-bold"),
                                html.Div("OCOV", className="text-center font-weight-bold"),
                                html.Div("NOCOV", className="text-center font-weight-bold"),
                                html.Div("PassFail", className="text-center font-weight-bold"),
                            ], className="d-flex justify-content-between mb-2")
                        ], width=12)
                    ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Tabs([
                                dbc.Tab([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Loading(
                                                id="loading-model-performance",
                                                type="circle",
                                                children=[dcc.Graph(id='model-performance-graph', className='dash-graph')]
                                            ),
                                            html.Div(id='model-metrics-container', className='metrics-container')
                                        ], width=12)
                                    ])
                                ], label="MODEL PERFORMANCE"),
                                
                                dbc.Tab([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Loading(
                                                id="loading-clustering",
                                                type="circle",
                                                children=[dcc.Graph(id='clustering-graph', className='dash-graph')]
                                            ),
                                            html.Div(id='clustering-metrics-container', className='metrics-container')
                                        ], width=8),
                                        dbc.Col([
                                            dcc.Graph(
                                                id='classification-pie',
                                                figure=create_pie_chart([100-high_fare_pct, high_fare_pct], ['Regular Fare', 'High Fare'], 
                                                                       "Fare Type Distribution"),
                                                config={'displayModeBar': False},
                                                className='dash-graph mb-3'
                                            ),
                                            html.Div([
                                                html.Div("Key Trip Statistics", className="font-weight-bold mb-2 text-center"),
                                                html.Div([
                                                    html.Div([f"Avg. Distance: {avg_distance:.2f} mi"], className="mb-1"),
                                                    html.Div([f"Avg. Fare: ${avg_fare:.2f}"], className="mb-1"),
                                                    html.Div([f"Max Fare: ${max_fare:.2f}"], className="mb-1"),
                                                    html.Div([f"High Fare %: {high_fare_pct:.1f}%"], className="mb-1"),
                                                ], className="metrics-container")
                                            ])
                                        ], width=4)
                                    ])
                                ], label="CLUSTERING"),
                                
                                dbc.Tab([
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Loading(
                                                id="loading-regression",
                                                type="circle",
                                                children=[dcc.Graph(id='regression-graph', className='dash-graph')]
                                            ),
                                            html.Div(id='regression-metrics-container', className='metrics-container')
                                        ], width=12)
                                    ])
                                ], label="REGRESSION"),
                                
                                dbc.Tab([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Div([
                                                html.H5("2-Fold Cross Validation", className="mb-3"),
                                                html.P("Compare models using 2-fold cross validation for more robust evaluation."),
                                                html.Button('RUN 2-FOLD CV', id='run-cv-button', className='btn btn-primary mb-3'),
                                                dcc.Loading(
                                                    id="loading-cv",
                                                    type="circle",
                                                    children=[
                                                        dcc.Graph(id='cv-comparison-graph', className='dash-graph'),
                                                        html.Div(id='cv-results-table', className='mt-3')
                                                    ]
                                                )
                                            ])
                                        ], width=12)
                                    ])
                                ], label="CROSS VALIDATION")
                            ])
                        ], width=12)
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Card([
                dbc.CardHeader("DATA EXPLORER"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='feature-x-dropdown',
                                placeholder="Select X Feature"
                            ),
                        ], width=6),
                        dbc.Col([
                            dcc.Dropdown(
                                id='feature-y-dropdown',
                                placeholder="Select Y Feature"
                            ),
                        ], width=6)
                    ], className="mb-3"),
                    dcc.Loading(
                        id="loading-scatter",
                        type="circle",
                        children=[dcc.Graph(id='scatter-explorer', className='dash-graph')]
                    )
                ])
            ])
        ], width=9)
    ]),
    
    html.Footer([
        html.P("NYC TAXI DATA ANALYTICS DASHBOARD © 2025", className="text-center text-muted mt-3 mb-2"),
        html.P("Designed by Your Name | Powered by Dash", className="text-center text-muted mb-3 small")
    ]),
    
    # Store components to keep track of the current state
    dcc.Store(id='knn-k-param-store', data=3),
    dcc.Store(id='rf-n-estimators-store', data=100),
    dcc.Store(id='gb-learning-rate-store', data=0.1),
    dcc.Store(id='dl-epochs-store', data=10),
    dcc.Store(id='cv-results-store', data=None)
], fluid=True, className='dash-container')

# Define the callbacks
@app.callback(
    [Output('model-params-container', 'children'),
     Output('knn-k-param-store', 'data', allow_duplicate=True),
     Output('rf-n-estimators-store', 'data', allow_duplicate=True),
     Output('gb-learning-rate-store', 'data', allow_duplicate=True),
     Output('dl-epochs-store', 'data', allow_duplicate=True)],
    [Input('model-dropdown', 'value')],
    [State('knn-k-param-store', 'data'),
     State('rf-n-estimators-store', 'data'),
     State('gb-learning-rate-store', 'data'),
     State('dl-epochs-store', 'data')],
    prevent_initial_call=True
)
def update_model_params(model, knn_k, rf_n, gb_lr, dl_ep):
    # Default returns - only update the value that changed
    if model == 'knn':
        return [
            html.Div([
                html.Label("Number of neighbors (k)"),
                dcc.Slider(3, 15, 2, value=knn_k, id='knn-k-param')
            ])
        ], dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif model == 'random_forest':
        return [
            html.Div([
                html.Label("Number of estimators"),
                dcc.Slider(50, 500, 50, value=rf_n, id='rf-n-estimators')
            ])
        ], dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif model == 'gradient_boosting':
        return [
            html.Div([
                html.Label("Learning rate"),
                dcc.Slider(0.01, 0.3, 0.01, value=gb_lr, id='gb-learning-rate')
            ])
        ], dash.no_update, dash.no_update, dash.no_update, dash.no_update
    elif model == 'deep_learning' and tensorflow_available:
        return [
            html.Div([
                html.Label("Epochs"),
                dcc.Slider(5, 50, 5, value=dl_ep, id='dl-epochs')
            ])
        ], dash.no_update, dash.no_update, dash.no_update, dash.no_update
    return [html.Div()], dash.no_update, dash.no_update, dash.no_update, dash.no_update

# Store slider values when they change
@app.callback(
    Output('knn-k-param-store', 'data'),
    Input('knn-k-param', 'value'),
    prevent_initial_call=True
)
def store_knn_value(value):
    return value if value is not None else 3

@app.callback(
    Output('rf-n-estimators-store', 'data'),
    Input('rf-n-estimators', 'value'),
    prevent_initial_call=True
)
def store_rf_value(value):
    return value if value is not None else 100

@app.callback(
    Output('gb-learning-rate-store', 'data'),
    Input('gb-learning-rate', 'value'),
    prevent_initial_call=True
)
def store_gb_value(value):
    return value if value is not None else 0.1

@app.callback(
    Output('dl-epochs-store', 'data'),
    Input('dl-epochs', 'value'),
    prevent_initial_call=True
)
def store_dl_value(value):
    return value if value is not None else 10

@app.callback(
    Output('clustering-params-container', 'children'),
    Input('clustering-dropdown', 'value')
)
def update_clustering_params(algorithm):
    if algorithm == 'kmeans':
        return [
            html.Label("Number of clusters"),
            dcc.Slider(2, 10, 1, value=3, id='kmeans-n-clusters')
        ]
    elif algorithm == 'dbscan':
        return [
            html.Label("Epsilon"),
            dcc.Slider(0.1, 5.0, 0.1, value=2.0, id='dbscan-eps'),
            html.Label("Min Samples", className='mt-2'),
            dcc.Slider(5, 50, 5, value=20, id='dbscan-min-samples')
        ]
    return []

@app.callback(
    [Output('feature-x-dropdown', 'options'),
     Output('feature-y-dropdown', 'options'),
     Output('feature-x-dropdown', 'value'),
     Output('feature-y-dropdown', 'value')],
    Input('model-performance-graph', 'figure')  # This is just a trigger
)
def set_features_dropdown_options(_):
    try:
        _, _, _, feature_cols, _ = get_features_and_targets()
        options = [{'label': col, 'value': col} for col in feature_cols]
        return options, options, 'trip_distance', 'fare_amount'
    except:
        # Return empty options if data loading fails
        return [], [], None, None

# Add new callback for 2-fold cross validation
@app.callback(
    [Output('cv-comparison-graph', 'figure'),
     Output('cv-results-table', 'children'),
     Output('cv-results-store', 'data')],
    Input('run-cv-button', 'n_clicks'),
    prevent_initial_call=True
)
def run_cross_validation(n_clicks):
    if n_clicks is None:
        return go.Figure(), html.Div(), None
    
    try:
        # Load data
        X, y_class, _, _, _ = get_features_and_targets()
        
        # Use a subset for faster execution in the dashboard
        subset_size = min(5000, X.shape[0])
        indices = np.random.choice(X.shape[0], subset_size, replace=False)
        X_subset = X[indices]
        y_subset = y_class[indices]
        
        # Run cross validation
        results_df, results_list = compare_models_2fold_cv(X_subset, y_subset, task="classification")
        
        # Create comparison graph
        model_names = [r['model_name'] for r in results_list]
        mean_scores = [r['mean_score'] for r in results_list]
        std_scores = [r['std_score'] for r in results_list]
        
        fig = go.Figure()
        
        # Add bars with error bars
        fig.add_trace(go.Bar(
            x=model_names,
            y=mean_scores,
            error_y=dict(
                type='data',
                array=std_scores,
                visible=True
            ),
            marker_color=colors['primary'],
            text=[f'{score:.3f}' for score in mean_scores],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='2-Fold Cross Validation Results',
            xaxis_title='Model',
            yaxis_title='Mean Accuracy',
            yaxis=dict(range=[0, 1.1], gridcolor=colors['grid']),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            margin=dict(l=40, r=40, t=40, b=40),
            showlegend=False
        )
        
        # Create results table
        table = dbc.Table.from_dataframe(
            results_df,
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="table-dark"
        )
        
        results_container = html.Div([
            html.H5("Cross Validation Results", className="mb-3"),
            html.P(f"Using {subset_size} samples for faster execution"),
            table
        ])
        
        return fig, results_container, results_list
        
    except Exception as e:
        error_fig = go.Figure()
        error_fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']}
        )
        return error_fig, html.Div(f"Error running cross validation: {str(e)}"), None

@app.callback(
    [Output('model-performance-graph', 'figure'),
     Output('model-metrics-container', 'children'),
     Output('gauge-chart', 'figure')],
    [Input('run-model-button', 'n_clicks'),
     Input('model-dropdown', 'value')],
    [State('knn-k-param-store', 'data'),
     State('rf-n-estimators-store', 'data'),
     State('gb-learning-rate-store', 'data'),
     State('dl-epochs-store', 'data')]
)
def run_and_display_model(n_clicks, model_type, knn_k, rf_n, gb_lr, dl_ep):
    ctx = dash.callback_context
    gauge_value = 50.0  # Default value
    
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'run-model-button':
        # Initialize with default values
        models = ['KNN', 'Logistic Regression', 'Random Forest', 'Gradient Boosting']
        if tensorflow_available:
            models.append('Deep Learning')
        accuracies = [0] * len(models)
        metrics_display = html.Div("Click 'RUN MODEL' to see performance metrics")
    else:
        # Load data
        X, y_class, _, _, _ = get_features_and_targets()
        X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
        
        # Initialize with zeros
        models = ['KNN', 'Logistic Regression', 'Random Forest', 'Gradient Boosting']
        if tensorflow_available:
            models.append('Deep Learning')
        accuracies = [0] * len(models)
        
        # Run selected model
        if model_type == 'knn':
            k = knn_k if knn_k is not None else 3
            knn = KNNFast(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies[0] = acc
            gauge_value = acc * 100
            metrics_display = html.Div([
                html.H5(f"KNN Results (k={k})"),
                html.P(f"Accuracy: {acc:.4f}")
            ])
        
        elif model_type == 'logistic':
            _, y_true, y_pred, acc = train_logistic_regression(X, y_class)
            accuracies[1] = acc
            gauge_value = acc * 100
            metrics_display = html.Div([
                html.H5("Logistic Regression Results"),
                html.P(f"Accuracy: {acc:.4f}")
            ])
        
        elif model_type == 'random_forest':
            n_estimators = rf_n if rf_n is not None else 100
            rf = train_random_forest(X_train, y_train, n_estimators=n_estimators)
            acc = rf.score(X_test, y_test)
            accuracies[2] = acc
            gauge_value = acc * 100
            metrics_display = html.Div([
                html.H5(f"Random Forest Results (n_estimators={n_estimators})"),
                html.P(f"Accuracy: {acc:.4f}")
            ])
        
        elif model_type == 'gradient_boosting':
            learning_rate = gb_lr if gb_lr is not None else 0.1
            gb = train_gradient_boosting(X_train, y_train, learning_rate=learning_rate)
            acc = gb.score(X_test, y_test)
            accuracies[3] = acc
            gauge_value = acc * 100
            metrics_display = html.Div([
                html.H5(f"Gradient Boosting Results (learning_rate={learning_rate})"),
                html.P(f"Accuracy: {acc:.4f}")
            ])
        
        elif model_type == 'deep_learning' and tensorflow_available:
            epochs = dl_ep if dl_ep is not None else 10
            model_dl = build_deep_learning_model(X.shape[1])
            model_dl.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            score = model_dl.evaluate(X_test, y_test, verbose=0)
            acc = score[1]
            accuracies[4] = acc
            gauge_value = acc * 100
            metrics_display = html.Div([
                html.H5(f"Deep Learning Results (epochs={epochs})"),
                html.P(f"Accuracy: {acc:.4f}")
            ])
    
    # Create the figure
    fig = go.Figure(data=[
        go.Bar(name='Accuracy', x=models, y=accuracies, marker_color=colors['primary'])
    ])
    
    fig.update_layout(
        title='Model Accuracy Comparison',
        xaxis_title='Model',
        yaxis_title='Accuracy',
        yaxis=dict(range=[0, 1.0], gridcolor=colors['grid']),
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        font={'color': colors['text']},
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Update gauge chart
    gauge_fig = create_gauge_chart(gauge_value, "Accuracy (%)", 0, 100)
    
    return fig, metrics_display, gauge_fig

@app.callback(
    [Output('clustering-graph', 'figure'),
     Output('clustering-metrics-container', 'children')],
    [Input('run-clustering-button', 'n_clicks'),
     Input('clustering-dropdown', 'value')],
    [State('kmeans-n-clusters', 'value'),
     State('dbscan-eps', 'value'),
     State('dbscan-min-samples', 'value')]
)
def run_and_display_clustering(n_clicks, algorithm, n_clusters=3, eps=2.0, min_samples=20):
    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'].split('.')[0] != 'run-clustering-button':
        # Return empty figure and message if button not clicked
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']}
        )
        return fig, html.Div("Click 'RUN CLUSTERING' to see results")
    
    try:
        # Load and prepare data
        X, _, _, _, _ = get_features_and_targets()
        
        # Scale before clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Sample data for faster processing
        indices = np.random.choice(X_scaled.shape[0], min(1000, X_scaled.shape[0]), replace=False)
        X_sample = X_scaled[indices]
        
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X_sample)
        
        # Run the selected clustering algorithm
        if algorithm == 'kmeans':
            n_clusters = n_clusters if n_clusters is not None else 3
            model, labels, silhouette = run_kmeans(X_sample, n_clusters=n_clusters)
            title = f"KMeans Clustering (n_clusters={n_clusters})"
            metrics = html.Div([
                html.H5(f"KMeans Results (n_clusters={n_clusters})"),
                html.P(f"Silhouette Score: {silhouette:.4f}")
            ])
        else:  # dbscan
            eps = eps if eps is not None else 2.0
            min_samples = min_samples if min_samples is not None else 20
            model, labels, silhouette = run_dbscan(X_sample, eps=eps, min_samples=min_samples)
            title = f"DBSCAN Clustering (eps={eps}, min_samples={min_samples})"
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            metrics = html.Div([
                html.H5(f"DBSCAN Results (eps={eps}, min_samples={min_samples})"),
                html.P(f"Number of clusters: {n_clusters}"),
                html.P(f"Silhouette Score: {silhouette if silhouette is not None else 'N/A'}")
            ])
            
        # Create the scatter plot
        fig = go.Figure()
        
        # Add trace for each cluster label
        for i in set(labels):
            mask = labels == i
            fig.add_trace(go.Scatter(
                x=X_vis[mask, 0], 
                y=X_vis[mask, 1],
                mode='markers',
                marker=dict(size=8),
                name=f'Cluster {i}' if i >= 0 else 'Noise'
            ))
            
        fig.update_layout(
            title=title,
            xaxis=dict(title='PCA Component 1', gridcolor=colors['grid']),
            yaxis=dict(title='PCA Component 2', gridcolor=colors['grid']),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            legend=dict(
                font=dict(color=colors['text']),
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig, metrics
    except Exception as e:
        # Return empty figure if clustering fails
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']}
        )
        return fig, html.Div(f"Error running clustering: {str(e)}")

@app.callback(
    [Output('regression-graph', 'figure'),
     Output('regression-metrics-container', 'children')],
    Input('model-performance-graph', 'figure')  # This is just a trigger
)
def update_regression_graph(_):
    try:
        # Load data and run regression
        X, _, y_reg, _, _ = get_features_and_targets()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        
        # Train regression model
        rf_reg = train_random_forest_regressor(X_train, y_train)
        mse, mae, r2 = evaluate_regression(rf_reg, X_test, y_test)
        
        # Make predictions for visualization
        y_pred = rf_reg.predict(X_test)
        
        # Sample for visualization
        sample_size = min(1000, len(y_test))
        indices = np.random.choice(len(y_test), sample_size, replace=False)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=y_test[indices],
            y=y_pred[indices],
            mode='markers',
            marker=dict(
                color=colors['primary'],
                size=8,
                opacity=0.7
            ),
            name='Predictions'
        ))
        
        # Add perfect prediction line
        fig.add_trace(go.Scatter(
            x=[min(y_test), max(y_test)],
            y=[min(y_test), max(y_test)],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color=colors['accent'], dash='dash')
        ))
        
        fig.update_layout(
            title='Random Forest Regression: Actual vs Predicted Values',
            xaxis=dict(title='Actual Fare Amount', gridcolor=colors['grid']),
            yaxis=dict(title='Predicted Fare Amount', gridcolor=colors['grid']),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            legend=dict(
                font=dict(color=colors['text']),
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        # Create metrics display
        metrics = html.Div([
            html.H5("Random Forest Regression Results"),
            html.P(f"Mean Squared Error: {mse:.2f}"),
            html.P(f"Mean Absolute Error: {mae:.2f}"),
            html.P(f"R² Score: {r2:.4f}")
        ])
        
        return fig, metrics
    except:
        # Return empty figure if regression fails
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']}
        )
        return fig, html.Div("Error running regression")

@app.callback(
    Output('scatter-explorer', 'figure'),
    [Input('feature-x-dropdown', 'value'),
     Input('feature-y-dropdown', 'value')]
)
def update_scatter_explorer(x_feature, y_feature):
    if not x_feature or not y_feature:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']}
        )
        return fig
    
    try:
        df = load_data()
        
        # Sample data for faster rendering
        sample_df = df.sample(min(1000, len(df)))
        
        # Create figure
        fig = go.Figure()
        
        # Create scatter plot with color based on high_fare
        if 'high_fare' in df.columns:
            for i, fare_type in enumerate(['Regular Fare', 'High Fare']):
                mask = sample_df['high_fare'] == i
                fig.add_trace(go.Scatter(
                    x=sample_df.loc[mask, x_feature],
                    y=sample_df.loc[mask, y_feature],
                    mode='markers',
                    marker=dict(
                        size=8,
                        opacity=0.7
                    ),
                    name=fare_type
                ))
        else:
            fig.add_trace(go.Scatter(
                x=sample_df[x_feature],
                y=sample_df[y_feature],
                mode='markers',
                marker=dict(
                    color=colors['primary'],
                    size=8,
                    opacity=0.7
                )
            ))
        
        fig.update_layout(
            title=f'{x_feature} vs {y_feature}',
            xaxis=dict(title=x_feature, gridcolor=colors['grid']),
            yaxis=dict(title=y_feature, gridcolor=colors['grid']),
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']},
            legend=dict(
                font=dict(color=colors['text']),
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    except:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font={'color': colors['text']}
        )
        return fig

if __name__ == '__main__':
    app.run(debug=True) 