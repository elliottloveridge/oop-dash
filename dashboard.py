# Dash imports
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
# Other imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Define constants (dropdown menu values)
VAR_OPTIONS = [
    {'label': 'Yes', 'value': True}
]
DATE_OPTIONS = [
    {'label': 'Daily', 'value': 'D'},
    {'label': 'Weekly', 'value': 'W'}
]
METRIC_OPTIONS = [
    {'label': 'Conversion', 'value': 'Conversion'},
    {'label': 'Total Price', 'value': 'Total Price'},
    {'label': 'Gross Profit', 'value': 'Gross Profit'},
    {'label': 'Net Profit', 'value': 'Net Profit'},
]
FILTER_OPTIONS = [
    {'label': 'Customer Age', 'value': 'Customer Age'},
    {'label': 'Credit Score', 'value': 'Credit Score'},
    {'label': 'Vehicle Value', 'value': 'Vehicle Value'},
    {'label': 'Vehicle Mileage', 'value': 'Vehicle Mileage'},
    {'label': 'Licence Length', 'value': 'Licence Length'}
]
# Temporary dropdown definition for filter category (dynamic dropdown that gets updated via callbacks)
category_options = [
    {'label': '-', 'value': '-'}
]


class Data:
    """Class to handle data and methods that act upon it.
    """

    def __init__(self, filepath: str) -> None:
        """Initialise DataFrame given a data `filepath` and perform data cleaning and transformations.

        Args:
            filepath (str): File path to data file.
        """
        self.df = pd.read_csv(filepath)
        self.rename_columns()
        self.remove_columns()
        self.convert_to_datetime()
        self.calculate_gross_profit()
        self.create_buckets()

    def rename_columns(self, mapper: dict[str, str] = {'Sale Indicator': 'Conversion', 'Profit': 'Net Profit'}) -> None:
        """Given a `mapper` dict, rename columns inplace for `self.df` DataFrame.

        Args:
            mapper (dict, optional): Column name mapping dictionary (format: {current_column_name: new_column_name}).
                Defaults to {'Sale Indicator': 'Conversion', 'Profit': 'Net Profit'}.
        """
        self.df = self.df.rename(columns=mapper)

    def remove_columns(self, columns: list[str] = ['Quote Number', 'Marital Status']) -> None:
        """Given a list of `columns`, remove them from `self.df` DataFrame.

        Args:
            columns (list[str], optional): List of columns to remove. Defaults to ['Quote Number', 'Marital Status'].
        """
        self.df = self.df.drop(columns, axis=1)

    def convert_to_datetime(self, date_col: str = 'Transaction Date', date_format: str = '%d/%m/%Y') -> None:
        """For `self.df`, convert a provided `date_col` of format `date_format` to type pandas datetime.

        Args:
            date_col (str, optional): Column to convert to datetime. Defaults to 'Transaction Date'.
            date_format (str, optional): Format of provided date column. Defaults to '%d/%m/%Y'.
        """
        self.df[date_col] = pd.to_datetime(
            self.df[date_col], format=date_format)

    def calculate_gross_profit(self) -> None:
        """For `self.df`, calculate Gross Profit as Net Profit + Tax.
        """
        #  # FIXME: Add try/catch (assumes Net Profit and Tax exist in the DataFrame)
        self.df['Gross Profit'] = self.df['Net Profit'] + self.df['Tax']

    def create_buckets(self, columns: list[str] = ['Customer Age', 'Credit Score', 'Licence Length'], n_buckets: int = 10) -> None:
        """Given a list of `columns`, bucket each columns values into `n_buckets` of equal size.

        For each column provided by `columns`, create a categorial column of `n_buckets` of equal size.
        Following this, reformat the column to be: 'lower_value-upper_value' with values rounded to 1d.p.

        Args:
            columns (list[str], optional): List of categorical columns in `self.df`. Defaults to ['Customer Age', 'Credit Score', 'Licence Length'].
            n_buckets (int, optional): Number of buckets to create for each column. Defaults to 10.
        """
        for col in columns:
            self.df[col] = pd.qcut(self.df[col], n_buckets)
            mapper = {
                i: f'{round(i.left, 1)}-{round(i.right, 1)}' for i in self.df[col].cat.categories
            }
            self.df[col] = self.df[col].cat.rename_categories(mapper)

    def group_data(self, group: list[str], filter_by: str, date_group: str = 'Transaction Date', date_type: str = None) -> pd.DataFrame:
        """Given a list of columns to group by, group `self.df` and return average values for a given metric (`filter_by`) within these groups.

        If a `date_type` is provided (supports Daily and Weekly), add `date_group` column to the groupby.

        Args:
            group (list[str]): List of columns to group by (exclude Transaction Date as handled separately).
            filter_by (str): Metric to filter on and return mean values for.
            date_group (str, optional): Date column to group by (not required). Defaults to 'Transaction Date'.
            date_type (str, optional): Date type to group `date_group` on (currently supports Daily (D) and Weekly (W)). Defaults to None.

        Returns:
            pd.DataFrame: Grouped DataFrame with mean values for a given metric (`filter_by`).
        """
        # If grouping by date then unpack the group columns and append a datetime grouper of frequency date_type
        if date_type:
            return self.df.groupby([*group, pd.Grouper(key=date_group, freq=date_type)])[filter_by].mean(numeric_only=True).reset_index()
        else:
            return self.df.groupby(group)[filter_by].mean(numeric_only=True).reset_index()

    def filter_data(self, df: pd.DataFrame, filter_by: str, category: str) -> pd.DataFrame:
        """Given a DataFrame, `df`, filter for rows where column `filter by` has a value equal to `category`.

        In the context of this application, `filter_data` is used to reduce the `df` to a single category bucket value (see `group_data`).

        Args:
            df (pd.DataFrame): DataFrame to filter.
            filter_by (str): Column to filter on.
            category (str): Category value to find within the `filter_by` column.

        Returns:
            pd.DataFrame: DataFrame filtered for a given category bucket value.
        """
        return df[df[filter_by] == category]

    def sort_values(self, df: pd.DataFrame, by: list[str] = ['Test Group', 'Transaction Date']) -> pd.DataFrame:
        """Given a DataFrame, `df`, sort its values `by` a list of provided columns in ascending order.

        Args:
            df (pd.DataFrame): DataFrame to sort.
            by (list[str], optional): List of columns to sort in order of sortation. Defaults to ['Test Group', 'Transaction Date'].

        Returns:
            pd.DataFrame: A sorted DataFrame.
        """
        return df.sort_values(by=by, ascending=True)

    def calculate_variance(self, df: pd.DataFrame, filter_by: str, variance_col: str = 'Test Group') -> pd.DataFrame:
        """Calculate the variance (difference) between matching rows for a given `variance_col` and metric (`filter_by`).

        In the context of this application, variance means Test Group A metric value - Test Group B metric value for matching conditions.

        Args:
            df (pd.DataFrame): A provided DataFrame comprised of a metric value and `variance_col` to compare values against.
            filter_by (str): Metric to calculate variance of.
            variance_col (str, optional): Column to compare difference with. Defaults to 'Test Group'.

        Returns:
            pd.DataFrame: A DataFrame of variance (A vs B) values for the defined metric.
        """
        # Create copy of df without metric values
        df_ = df.iloc[:, :-1].copy()
        # Only remove variance_col (Test Group) if other columns exist
        if len(df_.columns) > 1:
            df_ = df_.drop([variance_col], axis=1)
            # Remove duplpicate cases (A and B)
            df_ = df_.drop_duplicates()
        else:
            df_ = pd.DataFrame({'Test Group': ['A vs B']})

        # Define variance column as A - B
        df_[filter_by] = df[df[variance_col] == 'A'][filter_by].values - \
            df[df[variance_col] == 'B'][filter_by].values
        # Insert 'A vs B' as Test Group at position 0 to preserve correct sorting order
        if variance_col not in list(df_.columns.values):
            df_.insert(loc=0,
                       column=variance_col,
                       value='A vs B')

        return df_

    # def alternate_variance(self, df: pd.DataFrame, filter_by: str, variance_col: str = 'Test Group') -> pd.DataFrame:
    #     df_a = df[df[variance_col]=='A'].copy()
    #     df_b = df[df[variance_col] == 'B'].copy()
    #     df_a = df_a.drop([variance_col], axis=1)
    #     df_b = df_b.drop([variance_col], axis=1)
    #     df_ = pd.merge(df_a, df_b, on=list(df_a.columns.values)[:-1], how='inner')
    #     tmp = df_['Conversion_x'] - df_['Conversion_y']
    #     df_.drop(df_.columns[-2:], axis=1, inplace=True)
    #     df_['Conversion'] = tmp

    #     return df_

    def update_data(self, metric_value: str, date_value: str, filter_value: str, variance_value: bool) -> pd.DataFrame:
        """Given dropdown menu values, update the DataFrame to display figures and tables with.

        Calls `group_data`, `calculate_variance`, and `sort_values` methods
            depending on provided values via dashboard dropdown menu items.

        Args:
            metric_value (str): Metric dropdown value.
            date_value (str): Date dropdown value.
            filter_value (str): Filter dropdown value.
            variance_value (bool): Variance dropdown value.

        Returns:
            pd.DataFrame: A grouped and sorted DataFrame of `self.df`.
        """
        # Create group (default is Test Group - always grouped on)
        group = ['Test Group'] if filter_value is None else [
            'Test Group', filter_value]
        df = self.group_data(
            group=group, filter_by=metric_value, date_type=date_value)

        if variance_value:
            df = self.calculate_variance(df, filter_by=metric_value)

        # Get [:-1] columns (assumes metric value in final position) and set sort order as reverse
        # For df columns of Test Group, Transaction Date, Customer Age, Conversion it will pass
        # [Customer Age, Transaction Date, Test Group]
        columns_to_sort = list(df.columns[:(len(df.columns)-1)].values)[::-1]
        df = self.sort_values(df, by=columns_to_sort)

        return df


class Graph:
    """Class to handle creation and maintenance of figure/plot objects within dash and updates to their plot parameters.
    """

    def __init__(self, name: str, x: str, y: str = None, hue: str = 'Test Group', graph: str = 'bar') -> None:
        """Initialise common plot parameters.

        Args:
            name (str): Given name of a graph instance for reference as dash id.
            x (str): x-axis value for the figure.
            y (str, optional): y-axis value for the figure. Defaults to None.
            hue (str, optional): Color value for the figure (what figure values are split by). Defaults to 'Test Group'.
            graph (str, optional): Graph type to plot. Defaults to 'bar'.
        """
        self.name = name
        self.x = x
        self.y = y
        self.hue = hue
        self.graph = graph

    def create_figure(self, df: pd.DataFrame) -> list[object]:
        """Create a dcc.Graph object with id `self.name` and plot type line or bar.

        Args:
            df (pd.DataFrame): DataFrame to use for plot creation.

        Returns:
            list[object]: Graph object of figure and id.
        """
        if self.graph == 'bar':
            fig = self.create_barplot(df)
            # # # FIXME: Add values as labels on bars
            # fig.update_traces(textfont_size=12, textangle=0,
            #                   textposition='outside', cliponaxis=False)
        elif self.graph == 'line':
            fig = self.create_lineplot(df)

        return [
            dcc.Graph(
                figure=fig,
                id=self.name)
        ]

    def update_figure(self, metric_value: str, date_value: str, filter_value: str, variance_value: str) -> None:
        """Update defined figure parameters depending on dropdown value selection.

        Args:
            metric_value (str): Dropdown metric value.
            date_value (str): Date dropdown value.
            filter_value (str): Filter dropdown value.
            variance_value (str): Variance dropdown value.
        """
        # Use line plot for daily and weekly views
        self.graph = 'line' if date_value else 'bar'
        # Update y-axis for metric choice
        self.y = metric_value

        # Update x-axis for daily and weekly views (Test Group by default otherwise)
        if date_value:
            self.x = 'Transaction Date'
        elif filter_value:
            self.x = filter_value
        else:
            self.x = 'Test Group'

        # Update hue for double axis line plot
        if variance_value:
            self.hue = 'Test Group'
        elif date_value and filter_value:
            self.hue = filter_value
        else:
            self.hue = 'Test Group'

    def create_barplot(self, df: pd.DataFrame) -> object:
        """Given a DataFrame, `df`, create a grouped barplot using self figure parameters.

        Args:
            df (pd.DataFrame): DataFrame to use for plot data.

        Returns:
            object: Plotly barplot object.
        """
        return px.bar(
            data_frame=df,
            x=self.x,
            y=self.y,
            color=self.hue,
            barmode='group'
        )

    def create_lineplot(self, df: pd.DataFrame) -> object:
        """Given a DataFrame, `df`, create a lineplot using self figure parameters.

        Args:
            df (pd.DataFrame): DataFrame to use for plot data.

        Returns:
            object: Plotly lineplot object.
        """
        return px.line(
            data_frame=df,
            x=self.x,
            y=self.y,
            color=self.hue
        )

    def create_dualplot(self, df: pd.DataFrame, filter_by: str) -> object:
        """Create a dual-axis line plot for a given category (`filter_by`).

        When both date_value and filter_value are provided via dropdown menus, use a dual axis plot.
        This allows for Transaction Date as an x-axis and Test Group as a hue for a given category.

        Args:
            df (pd.DataFrame): DataFrame to use for plot data.
            filter_by (str): Category to filter data on.

        Returns:
            object: Plotly graph object.
        """
        fig = go.Figure(
            # Define two line plots using hard-coded reference for each Test Group.
            data=[
                go.Scatter(name='A',
                           x=df[df['Test Group'] == 'A']['Transaction Date'],
                           y=df[df['Test Group'] == 'A'][filter_by],
                           yaxis='y',
                           mode='lines',
                           offsetgroup=1
                           ),
                go.Scatter(name='B',
                           x=df[df['Test Group'] == 'B']['Transaction Date'],
                           y=df[df['Test Group'] == 'B'][filter_by],
                           yaxis='y2',
                           mode='lines',
                           offsetgroup=2
                           ),
            ],
            # Update as dual axis plot
            layout={
                'yaxis': {'title': 'Test Group A'},
                'yaxis2': {'title': 'Test Group B', 'overlaying': 'y', 'side': 'right'}
            }
        )

        # Scale dual axes to match using min/max value from metric data
        if not df[filter_by].empty:
            value_range = [min(df[filter_by]), max(df[filter_by])]
            fig.update_layout(yaxis=dict(range=value_range),
                              yaxis2=dict(range=value_range))
            fig.update_layout(yaxis2=dict(scaleanchor='y'))

        return fig


class Element:
    """Class to handle creation and maintenance of dashboard elements (Dropdown and Table).
    """

    def __init__(self, name: str, label: str) -> None:
        """Initialise element with a given `name` (dash id) and `label` for display.

        Args:
            name (str): Dash id for reference of dashboard element.
            label (str): Readable label for display of element.
        """
        self.name = name
        self.label = label

    def create_dropdown(self, options: dict[str, str], default_value: str = None, clearable: bool = True) -> list[object]:
        """Create a dash dropdown element using defined `options`.

        Args:
            options (dict[str, str]): A dictionary of dropdown menu items.
            default_value (str, optional): Default value for a dropdown. Defaults to None.
            clearable (bool, optional): Boolean to allow for removal of dropdown value. Defaults to True.

        Returns:
            list[object]: Dash dropdown object and HTML label for display.
        """
        if default_value:
            dropdown = dcc.Dropdown(
                id=self.name,
                options=options,
                value=default_value,
                className='dropdown',
                clearable=clearable
            )
        else:
            dropdown = dcc.Dropdown(
                id=self.name,
                options=options,
                className='dropdown',
                clearable=clearable
            )

        return [
            html.Label(self.label),
            dropdown
        ]

    def create_table(self, df: pd.DataFrame) -> object:
        """Create a dash Table for a given DataFrame `df`.

        Create a data table to display `df` on the dashboard.
        Variables not defined in from_dataframe method used are for customising output layout.

        Args:
            df (pd.DataFrame): DataFrame to use for table data.

        Returns:
            object: Dash Table object.
        """
        return dbc.Table.from_dataframe(df, id=self.name, striped=True, bordered=True, hover=True)

    def update_dropdown(self, df: pd.DataFrame, filter_value: str) -> list[dict[str, str]]:
        """Update dynamic dropdown menu items using unique values from a `df` column (`filter_value`).

        Args:
            df (pd.DataFrame): DataFrame to update dropdown with.
            filter_value (str): Column to get unique values from.

        Returns:
            list[dict[str, str]]: Dictionary of unique. dropdown menu options.
        """
        # Update category_dropdown dynamic options dict
        items = list(df[filter_value].unique())
        #  # FIXME: add sort for items list
        return [{'label': i, 'value': i} for i in items]

    def update_visibility(self, date_value: str, filter_value: str) -> dict[str, str]:
        """Update visbility of an element by setting its container display style.

        Used within this application to show category_dropdown only when both `date_value` and `filter_value` have values.
        Requires the use of an Output callback with reference to the dash id.

        Args:
            date_value (str): Date dropdown value.
            filter_value (str): Filter dropdown value.

        Returns:
            dict[str, str]: Dictionary to update style of dropdown object.
        """
        if date_value and filter_value:
            return {'display': 'block'}
        else:
            return {'display': 'none'}


class Dashboard:
    """Class to handle dash update callbacks and dashboard layout configuration.
    """

    def __init__(self, app: object, filepath: str = 'data/data.csv') -> None:
        """Initialises `data` file, applicaiton instance, and dashboard elements.

        Element and Graph objects are created using composition for each required dashboard object.
        `self.app.callback` defines input values from dashboard interaction and dependent output dashboard changes (e.g. figure).

        Args:
            app (object): Dash app instance.
            filepath (str, optional): File path of dataset. Defaults to 'data/data.csv'.
        """
        self.data = Data(filepath)
        self.app = app
        # Create dashboard elements (dropdowns, figure, table)
        self.metric_dropdown = Element('metric_dropdown', 'Metric')
        self.date_dropdown = Element('date_dropdown', 'Date Type')
        self.filter_dropdown = Element('filter_dropdown', 'Filter')
        self.category_dropdown = Element('category_dropdown', 'Category')
        self.var_dropdown = Element('variance_dropdown', 'Variance')
        self.figure = Graph('figure', 'Test Group', 'Conversion')
        self.table = Element('table', 'Data Table')
        # Define dashboard callback requests
        self.app.callback(
            Output('figure', 'figure'),
            Output('table', 'children'),
            Output('category_dropdown', 'options'),
            Output('category_dropdown', 'style'),
            Input('metric_dropdown', 'value'),
            Input('date_dropdown', 'value'),
            Input('filter_dropdown', 'value'),
            Input('category_dropdown', 'value'),
            Input('variance_dropdown', 'value')
        )(self.update_dashboard)

    def update_dashboard(self, metric_value: str, date_value: str, filter_value: str, category_value: str, variance_value: str):
        """Update data and callback outputs given input callback values.

        Args:
            metric_value (str): Metric dropdown value.
            date_value (str): Date dropdown value.
            filter_value (str): Filter dropdown value.
            category_value (str): Category dropdown value.
            variance_value (str): Variance dropdown value.

        Returns:
            figure (object): Figure dash object for plot display.
            table (object): Table dash object for table display.
            category_options (dict): Dictionary of category dropdown options.
            display (dict): Dictionary to update category dropdown display style.

        """
        # Update parameters of figure
        self.figure.update_figure(
            metric_value, date_value, filter_value, variance_value)
        # Update data (group and sort)
        df = self.data.update_data(
            metric_value, date_value, filter_value, variance_value)

        if date_value and filter_value:
            # Update cateogry_dropdown dynamic options and filter based on category_dropdown value
            category_options = self.category_dropdown.update_dropdown(
                df, filter_value)
            df = self.data.filter_data(df, filter_value, category_value)

            # Use line plot vs dualplot if variance_value = 'Yes'
            if variance_value == True:
                figure = self.figure.create_lineplot(df)
            else:
                figure = self.figure.create_dualplot(df, metric_value)
        else:
            # Redefine default category_dropdown values if unselected
            category_options = [
                {'label': '-', 'value': '-'}
            ]
            # Update figure object for case of non-dual axis plot
            if self.figure.graph == 'line':
                figure = self.figure.create_lineplot(df)
            elif self.figure.graph == 'bar':
                figure = self.figure.create_barplot(df)

        # Update category_dropdwn to be shown/hidden
        display = self.category_dropdown.update_visibility(
            date_value, filter_value)
        # Redefine table using updated dataset
        table = self.table.create_table(df)

        return figure, table, category_options, display

    def layout(self):
        """Update dashbaord layout using dbc and created dropdowns, figures, and tables.
        """
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1('Pricing Dashboard')
                ],
                    width='auto')
            ],
                justify='center'
            ),
            # Define dropdown menu items usign constants
            dbc.Row([
                dbc.Col(
                    self.metric_dropdown.create_dropdown(
                        options=METRIC_OPTIONS, default_value=METRIC_OPTIONS[0]['label'], clearable=False),
                    width=3),
                dbc.Col(
                    self.date_dropdown.create_dropdown(options=DATE_OPTIONS),
                    width=3),
                dbc.Col(
                    self.filter_dropdown.create_dropdown(
                        options=FILTER_OPTIONS),
                    width=3),
                dbc.Col(
                    self.var_dropdown.create_dropdown(options=VAR_OPTIONS),
                    width=3)
            ]),
            # Define hidden category_dropdown
            html.Div([
                dbc.Row([
                    dbc.Col(
                        self.category_dropdown.create_dropdown(
                            options=category_options, default_value=category_options[0]['label'], clearable=False),
                        width=12)
                ]),],
                {'display': 'none'}
            ),
            dbc.Row([
                dbc.Col(
                    self.figure.create_figure(self.data.df)
                )
            ]),
            dbc.Row([
                dbc.Col(
                    self.table.create_table(self.data.df.head())
                )
            ])
        ])
