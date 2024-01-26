#importing the required libraries
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objs as go
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import Matern
from flask import Flask

#initialsing a flask application
app = Flask(__name__)

#defining a function for the black-box model
def black_box_function(C, X_train_scaled, y_train):
    model = SVR(C=C)
    model.fit(X_train_scaled, y_train)
    f = model.score(X_train_scaled, y_train)
    return f

#loading data from a pickle file
with open("protein_data.pkl", "rb") as pickle_file:
    protein_data = pickle.load(pickle_file)

#defining function - for data preprocessing
def preprocess_data(protein_data):
    # remove the null values for the reliable target
    protein_data1 = protein_data.dropna(axis=0, subset=['reliable_target_protein_super'])
    # fill the NA values by 0
    protein_data1 = protein_data1.fillna(0)

    # forming the target-data and feature-data
    feature = protein_data1.drop(
        ['chembl_id', 'target_id', 'component_synonym', 'type_synonym', 'protein_class_desc', 'pref_name',
         'EC_super_class', 'EC_super_class_name', 'protein_family', 'protein_super_family', 'EC_name'], axis=1)
    target = protein_data1['target_id']

    # converting the categorical features to dummy variables
    feature = pd.get_dummies(feature,
                             columns=['reliable_target_EC', 'reliable_target_protein_desc', 'reliable_target_EC_super',
                                      'reliable_target_protein_super'], drop_first=True)

    return feature, target



# initializing a Dash application
#defining the layout and components of the interface 
app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')
app.layout = html.Div(style={'backgroundColor': 'black', 'color': 'red', 'textAlign': 'center', 'padding': '20px'},
                      children=[
                          html.H1("MolOpt Explore App", style={'color': 'darkorange', 'marginBottom': '30px'}),

                          html.Div(style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'center',
                                          'alignItems': 'center', 'marginBottom': '20px'},
                                   children=[
                                       html.Label('Enter C Value:',
                                                  style={'marginRight': '10px', 'color': 'darkorange'}),
                                       dcc.Input(id='c-value', type='number', value=1.0, step=0.1,
                                                 style={'backgroundColor': 'darkgray', 'color': 'white'}),
                                       html.Label('Select Acquisition Function:',
                                                  style={'marginLeft': '20px', 'marginRight': '10px',
                                                         'color': 'darkorange'}),
                                       dcc.Dropdown(
                                           id='acquisition-function',
                                           options=[
                                               {'label': 'Expected Improvement (EI)', 'value': 'ei'},
                                               {'label': 'Probability of Improvement (PI)', 'value': 'pi'},
                                               {'label': 'Upper Confidence Bound (UCB)', 'value': 'ucb'}
                                           ],
                                           value='ei',
                                           style={'width': "200px"},
                                       ),
                                       html.Button('Generate Charts', id='generate-button',
                                                   style={'backgroundColor': 'darkorange', 'color': 'black',
                                                          'marginLeft': '20px'}),
                                   ]),

                          html.Div(
                              style={'display': 'flex', 'flexDirection': 'row', 'backgroundColor': 'rgba(0, 0, 0, 0)'},
                              children=[
                                  dcc.Graph(id='distribution-histogram',
                                            style={'width': '33%', 'margin-left': '10px', 'margin-top': '100px',
                                                   'background': 'transparent','paper_bgcolor':'black'}),
                                  dcc.Graph(id='correlation-heatmap',
                                            style={'margin-left': '30px', 'margin-top': '100px', 'width': '60%',
                                                   'background': 'transparent','paper_bgcolor':'black'}),
                              ]),

                          dcc.Graph(id='bayesian-convergence',
                                    style={'margin-left': '10px', 'width': '95%', 'margin-right': '50px',
                                           'margin-top': '50px', 'background': 'transparent','paper_bgcolor':'black'}),
                      ])

#creating call-backs for creation of histograms and correlation heatmaps
#this changes depending upon the user-inputs
@app.callback(
    [Output('distribution-histogram', 'figure'),
     Output('correlation-heatmap', 'figure')],
    [Input('generate-button', 'n_clicks'),
     Input('c-value', 'value')]
)
def generate_charts(n_clicks, c_value):
    if n_clicks is None:
        return {}, {}

    # pre-processing the data
    feature, target = preprocess_data(protein_data)

    # calculating the charts
    distribution_histogram = go.Figure(
        data=[go.Histogram(x=target, nbinsx=20)],
        layout=go.Layout(title='Distribution of Target Variable', paper_bgcolor='black')
    )

    correlation_matrix = feature.corr()
    correlation_heatmap = go.Figure(
        data=[go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.columns)],
        layout=go.Layout(title='Correlation Matrix Heatmap', paper_bgcolor='black')
    )

    return distribution_histogram, correlation_heatmap


#creating call-back for generating bayesian optimization curve
#this changes again depending upon user-inputs and selected aquisition functions
@app.callback(
    Output('bayesian-convergence', 'figure'),
    [Input('generate-button', 'n_clicks'),
     Input('c-value', 'value'),
     Input('acquisition-function', 'value')]
)
def generate_convergence_curve(n_clicks, c_value, acquisition_function):
    if n_clicks is None:
        return {}

    feature, target = preprocess_data(protein_data)
    feature_scaled = MinMaxScaler().fit_transform(feature)

    # Process of Bayesian Optimisation(BO)for the chosen acquisition function
    pbounds = {'C': (0.1, 10.0)}
    kernel = Matern(length_scale=1.0, nu=2.5)

    if acquisition_function == 'ei':
        af = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)
    elif acquisition_function == 'pi':
        af = UtilityFunction(kind="poi", kappa=2.5, xi=0.0)
    elif acquisition_function == 'ucb':
        af = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    optimizer = BayesianOptimization(
        f=lambda C: black_box_function(c_value, feature_scaled, target),
        pbounds=pbounds, verbose=2, random_state=4
    )

    optimizer._gp.set_params(kernel=kernel)  # Update the kernel based on acquisition function

    optimizer.maximize(
        init_points=5, n_iter=10,
        bounds_transformer=None, acq_optimizer="lbfgs",
        n_jobs=-1
    )
    results = optimizer.res

    # extracting the C values from the optimizer's results
    c_values = [res['params']['C'] for res in results]

    # creating a convergence curve graph for the selected or chosen acquisition function
    convergence_curve = go.Figure(
        data=[
            go.Scatter(
                x=list(range(1, len(results) + 1)),
                y=[res['target'] for res in results],
                mode='lines+markers',
                name=acquisition_function
            ),
            go.Scatter(
                x=list(range(1, len(results) + 1)),
                y=c_values,
                mode='lines',
                name='C Value',
                yaxis='y2'
            )
        ],
        layout=go.Layout(
            title='Bayesian Optimization Convergence Curve',
            paper_bgcolor='black',
            yaxis_type='log',
            yaxis=dict(title='Target'),
            yaxis2=dict(title='C Value', overlaying='y', side='right')
        )
    )

    return convergence_curve

#executing script directly to run the dash app
if __name__ == '__main__':
    app.run(debug=True)
