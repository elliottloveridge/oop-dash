import dash
import dash_bootstrap_components as dbc
from dashboard import Dashboard


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dashboard = Dashboard(app=app, filepath='data/data.csv')
dashboard.layout()

if __name__ == '__main__':
    app.run_server(debug=True)
