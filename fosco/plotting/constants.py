from plotly.graph_objs import Figure as plotlyFigure
from matplotlib.figure import Figure as mplFigure

FigureType = plotlyFigure | mplFigure

DOMAIN_COLORS = {
    "init": "blue",
    "unsafe": "red",
    "lie": "green",
    "robust": "green",
}
