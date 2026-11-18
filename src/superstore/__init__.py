"""Product sales analytics and weekly revenue forecasting.

The package is organised as a pipeline: ``data`` -> ``clean`` -> ``features`` ->
``eda`` / ``timeseries`` / ``segments`` / ``anomalies`` -> ``forecast`` ->
``report``, with ``dashboard`` reading only the committed artefacts.
"""

__version__ = "0.1.0"
