import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px

# Read the dataset
data = pd.read_csv("NIFTY50_all.csv")

# Convert the "Date" column to datetime format
data["Date"] = pd.to_datetime(data["Date"], format='%Y-%m-%d')

# Create year and month columns for analysis
data['year'] = data['Date'].dt.year
data["month"] = data["Date"].dt.month

# Visualize data (if needed)
# ...

# List of unique symbols (companies)
symbols = data['Symbol'].unique()

# Create a dictionary to store Prophet models and forecasts for each company
company_models = {}
company_forecasts = {}

for symbol in symbols:
    company_data = data[data['Symbol'] == symbol]

    # Prepare data for Prophet
    forecast_data = company_data.rename(columns={"Date": "ds", "Turnover": "y"})

    # Create and train the Prophet model for the company
    model = Prophet()
    model.fit(forecast_data)

    # Store the model in the dictionary
    company_models[symbol] = model

# Option to select a company for prediction
while True:
    print("Available companies:")
    for i, symbol in enumerate(symbols):
        print(f"{i + 1}. {symbol}")

    choice = input("Enter the number of the company you want to predict (or 'exit' to quit): ")

    if choice.lower() == 'exit':
        break

    try:
        choice = int(choice)
        if 1 <= choice <= len(symbols):
            selected_symbol = symbols[choice - 1]

            # Make future dataframe for predictions (adjust the number of days as needed)
            future = company_models[selected_symbol].make_future_dataframe(periods=365)

            # Predict using the model
            forecast = company_models[selected_symbol].predict(future)

            # Plot the forecast for the selected company
            fig = plot_plotly(company_models[selected_symbol], forecast)
            fig.update_layout(title=f"Turnover Prediction for {selected_symbol}")
            fig.show()
        else:
            print("Invalid choice. Please select a valid company number.")
    except ValueError:
        print("Invalid input. Please enter a valid company number or 'exit' to quit.")
