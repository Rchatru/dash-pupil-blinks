import pandas as pd
import numpy as np
# Import Dash components
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.fftpack import fft
from scipy.signal import hann
from scipy.signal import spectrogram
# Import Plotly for graphing
import plotly.graph_objects as go
import boto3



#df = pd.read_parquet('pupil_blinks.parquet.gzip')
df = pd.read_parquet('s3://render-dashboard-pupil/pupil_blinks.parquet.gzip')

# Create a Dash app
app = dash.Dash(__name__)
server = app.server

fs = 300 # Hz

# App layout
app.layout = html.Div([
    dcc.Dropdown(
        id='individual-selector',
        options=[{'label': i, 'value': i} for i in df['id'].unique()],
        value=df['id'].unique()[0]
    ),
    dcc.Graph(id='time-series-graph'),
    dcc.Graph(id='fft-graph'),
    dcc.Graph(id='spectrogram-graph')
])

# Callback to update time-series graph based on selected individual
@app.callback(
    Output('time-series-graph', 'figure'),
    [Input('individual-selector', 'value')]
)


def update_time_series(selected_id):
    sub_df = df[df['id'] == selected_id]
    
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02, 
                        specs=[[{"secondary_y": False}],
                                [{"secondary_y": False}],
                                [{"secondary_y": True}],
                                [{"secondary_y": False}],
                                [{"secondary_y": False}],
                                [{"secondary_y": False}]
                                ])

    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['PupilLeft'], mode='lines', name='PupilLeft'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['FilteredPupilLeft'], mode='lines', name='FilteredPupilLeft'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['PupilRight'], mode='lines', name='PupilRight'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['FilteredPupilRight'], mode='lines', name='FilteredPupilRight'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['VelLeft'], mode='lines', name='VelLeft'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['VelRight'], mode='lines', name='VelRight'),
                  row=2, col=1)
    # fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['AccLeft'], mode='lines', name='AccLeft'),
    #               row=2, col=1)
    # fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['AccRight'], mode='lines', name='AccRight'),
    #               row=2, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['GazePointLeftX (ADCSmm)'], mode='lines', name='GazePointLeftX'),
                  row=3, col=1, secondary_y = False)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['GazePointRightX (ADCSmm)'], mode='lines', name='GazePointRightX'),
                  row=3, col=1, secondary_y = False)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['FilteredGazePointLeftX (ADCSmm)'], mode='lines', name='FilteredGazePointLeftX'),
                  row=3, col=1, secondary_y = False)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['GazePointLeftY (ADCSmm)'], mode='lines', name='GazePointLeftY'),
                  row=3, col=1, secondary_y = True)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['GazePointRightY (ADCSmm)'], mode='lines', name='GazePointRightY'),
                  row=3, col=1, secondary_y = True)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['FilteredGazePointLeftY (ADCSmm)'], mode='lines', name='FilteredGazePointLeftY'),
                  row=3, col=1, secondary_y = True)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['EyePosLeftX (ADCSmm)'], mode='lines', name='EyePosLeftX'),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['EyePosRightX (ADCSmm)'], mode='lines', name='EyePosRightX'),
                  row=4, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['EyePosLeftY (ADCSmm)'], mode='lines', name='EyePosLeftY'),
                  row=5, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['EyePosRightY (ADCSmm)'], mode='lines', name='EyePosRightY'),
                  row=5, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['EyePosLeftZ (ADCSmm)'], mode='lines', name='EyePosLeftZ'),
                  row=6, col=1)
    fig.add_trace(go.Scatter(x=sub_df['RecordingTimestamp'], y=sub_df['EyePosRightZ (ADCSmm)'], mode='lines', name='EyePosRightZ'),
                  row=6, col=1)
    


    # Pre-calcula los inicios y finales de los parpadeos
    blink_ranges = sub_df[sub_df['BlinkIndex'] != 0].groupby('BlinkIndex')['RecordingTimestamp'].agg(['min', 'max'])
    # Itera sobre los rangos de parpadeo precalculados
    for _, (start, end) in blink_ranges.iterrows():
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="LightSalmon", opacity=0.6,
            line_width=0,
            row=1, col=1
        )
    
    data_loss_ranges = sub_df[sub_df['DataLossIndex'] != 0].groupby('DataLossIndex')['RecordingTimestamp'].agg(['min', 'max'])
    # Itera sobre los rangos de parpadeo precalculados
    for _, (start, end) in data_loss_ranges.iterrows():
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="LightSlateGrey", opacity=0.6,
            line_width=0,
            row=1, col=1
        )

    great_loss_ranges = sub_df[sub_df['GreatLossIndex'] != 0].groupby('GreatLossIndex')['RecordingTimestamp'].agg(['min', 'max'])
    # Itera sobre los rangos de parpadeo precalculados
    for _, (start, end) in great_loss_ranges.iterrows():
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="DarkSlateGrey", opacity=0.6,
            line_width=0,
            row=1, col=1
        )
        
   
    # Update xaxis and yaxis properties
    # fig.update_xaxes(title_text='RecordingTimestamp', rangeslider_visible=False, row=1, col=1)
    fig.update_yaxes(title_text='',row=1, col=1, title_font=dict(size=8),title_standoff=25)
    fig.update_yaxes(title_text='',row=2, col=1, title_font=dict(size=8),title_standoff=5)
    fig.update_yaxes(title_text='',row=3, col=1, title_font=dict(size=8),title_standoff=5)
    fig.update_yaxes(title_text='',row=4, col=1, title_font=dict(size=8),title_standoff=10)
    fig.update_yaxes(title_text='',row=5, col=1, title_font=dict(size=8),title_standoff=10)
    fig.update_yaxes(title_text='',row=6, col=1, title_font=dict(size=8),title_standoff=10)
    fig.update_xaxes(title_text='RecordingTimestamp [ms]', row=6, col=1)

    # Títulos del eje Y de forma manual para correcta alineación
    titles = ['Pupil Size [mm]', 'Pupil Velocity [mm/s]', 'GazePoint [mm]', 'EyePosX [mm]', 'EyePosY [mm]', 'EyePosZ [mm]']
    number_of_y_axes = len(titles)  # Asumiendo que tienes un título para cada eje Y
    height_per_subplot = 1 / number_of_y_axes  # Ajusta si hay espacio entre subplots

    for i, title in enumerate(titles, start=1):
        fig.add_annotation(
            text=title,  # Título del eje Y
            xref="paper", yref="paper",
            x=-0.03, y=height_per_subplot * (number_of_y_axes - i) + (height_per_subplot / 2),
            showarrow=False,
            textangle=-90,
            xanchor="center",
            yanchor="middle",
            font=dict(size=10)
        )
    
    # Update layout and title
    fig.update_layout(template="plotly_white", height=850, title_text='Pupil Size and Velocity vs RecordingTimestamp')
    fig.update_xaxes(spikemode='across+marker',spikethickness=1)
    fig.update_traces(xaxis="x6")
    return fig

# Callback to update FFT and Spectrogram based on time range selected in time-series graph
@app.callback(
    [Output('fft-graph', 'figure'),
     Output('spectrogram-graph', 'figure')],
    [Input('individual-selector', 'value'),
     Input('time-series-graph', 'relayoutData')]
)
def update_fft_and_spectrogram(selected_id, relayoutData):
    sub_df = df[df['id'] == selected_id]
    fft_figure = dash.no_update
    spectrogram_figure = dash.no_update

    if relayoutData is not None:
        if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
            start = float(relayoutData['xaxis.range[0]'])
            end = float(relayoutData['xaxis.range[1]'])
            plot_data = sub_df[(sub_df['RecordingTimestamp'] >= start) & (sub_df['RecordingTimestamp'] <= end)]

            # FFT graph
            fft_result = np.fft.fft(plot_data['FilteredPupilLeft'])
            frequencies = np.fft.fftfreq(len(fft_result), d=1/fs)
            fft_figure = {
                'data': [go.Scatter(x=frequencies[1:len(frequencies)//2], y=np.abs(fft_result)[1:len(frequencies)//2], mode='lines')],
                'layout': go.Layout(title='FFT of selected range', xaxis={'title': 'Frequency'}, yaxis={'title': 'Amplitude'})
            }

            # Spectrogram graph
            f, t, Sxx = spectrogram(plot_data['FilteredPupilLeft'], fs=fs)
            spectrogram_figure = {
                'data': [go.Heatmap(x=t, y=f, z=Sxx)],
                'layout': go.Layout(title='Spectrogram', xaxis={'title': 'Time'}, yaxis={'title': 'Frequency'})
            }

    return fft_figure, spectrogram_figure

# Run the app
#if __name__ == '__main__':
#    app.run(debug=True, use_reloader=False, port=8050)

if __name__ == '__main__':
    app.run_server(debug=False)
