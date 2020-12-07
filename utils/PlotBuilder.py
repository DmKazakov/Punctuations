import numbers

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go


class PlotBuilder:
    def __init__(self, title: str, x_title: str, y_title: str):
        init_notebook_mode(connected=True)
        self.figure = go.Figure()
        self.figure.update_layout(
            title=title,
            xaxis_title=x_title,
            yaxis_title=y_title
        )

    def add_trace(self, x: [numbers.Number], y: [numbers.Number], name: str):
        self.figure.add_trace(go.Scatter(x=x, y=y, name=name))

    def write_to_file(self, file: str):
        self.figure.write_html(file)

    def plot_offline(self):
        iplot(self.figure)