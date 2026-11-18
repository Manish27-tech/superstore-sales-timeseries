# Data dictionary

Columns of the cleaned order table (`data/processed/orders.csv.gz`, git-ignored
but reproducible via `python run.py data`).

## Source columns (unchanged)

| Column | Type | Description |
|---|---|---|
| Row ID | int | Source row number |
| Order ID | str | `XX-YYYY-NNNNNN` |
| Order Date | date | Authoritative order date |
| Ship Date | date | Ship date |
| Ship Mode | str | Standard Class, Second Class, First Class, Same Day |
| Customer ID / Name | str | |
| Segment | str | Consumer, Corporate, Home Office |
| Country, City, State, Postal Code, Region | str | Geography |
| Product ID, Product Name, Category, Sub-Category | str | Product |
| Sales | float | Line revenue (USD) |
| Quantity | int | Units |
| Discount | float | 0.0 to 0.8 |
| Profit | float | Line profit (USD) |

## Derived columns

| Column | Type | Description |
|---|---|---|
| Order Year | int | From Order Date |
| Order Month | int | 1-12 |
| Order Quarter | int | 1-4 |
| Order Weekday | str | Monday ... Sunday |
| Weekday Index | int | 0 = Monday |
| Week Start | date | Monday of the order's week |
| Month Start | date | First day of the order's month |
| Year Month | str | `YYYY-MM` |
| Day of Year | int | 1-366 |
| Ship Lag Days | int | Ship Date - Order Date |
| Margin | float | Profit / Sales |
| Unit Price | float | Sales / Quantity |
| Is Return | int | 1 if Profit < 0 |
| Has Discount | int | 1 if Discount > 0 |
| Discount Band | str | 0% / 1-10% / 11-20% / 21-40% / 41%+ |
| Id Year Mismatch | int | 1 if the year in Order ID disagrees with Order Date |

## Aggregates

- `daily.csv`: `date, revenue, orders` (zero-filled).
- `weekly.csv`: `week_start, revenue, profit, units, orders, lines, margin, aov`.
- `monthly.csv`: `month_start, revenue, profit, units, orders, margin, aov`.
- `monthly_category.csv`: `month_start, Furniture, Office Supplies, Technology`.
- `weekday.csv`: `weekday, revenue, orders, lines`.
- `products.csv`: one row per product with level, growth, volatility, ABC class and segment.
- `forecast.csv`: held-out weeks with actual, predicted, horizon and interval.
- `backtest.csv`: walk-forward RMSE per fold and model.
- `anomalies.csv`: flagged weeks with their explanation.
- `kpis.json`: everything the dashboard header and about tab need.
- `dimensions.json`: filter values for the dashboard.
