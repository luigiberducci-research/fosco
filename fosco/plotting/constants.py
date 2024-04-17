from plotly.graph_objs import Figure
import matplotlib as mpl

FigureType = Figure | mpl.figure.Figure

DOMAIN_COLORS = {
    "init": "blue",
    "unsafe": "red",
    "lie": "green",
    "robust": "green",
}
