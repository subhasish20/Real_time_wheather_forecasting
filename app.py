import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from datetime import timedelta, datetime
import pytz

# ========== Utility Functions ==========

def get_current_weather(city):
    API_KEY = 'a5e8713c6700bfce420cd38b90e2122b'
    BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'
    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        return None

    data = response.json()
    if 'weather' not in data or not data['weather']:
        st.error("Weather data not found.")
        return None

    return {
        'city': data.get('name', 'N/A'),
        'current_temp': round(data['main']['temp']),
        'min_temp': data['main']['temp_min'],
        'max_temp': data['main']['temp_max'],
        'humidity': data['main']['humidity'],
        'pressure': data['main'].get('pressure', 1013),
        'wind_speed': data['wind']['speed'],
        'wind_deg': data['wind'].get('deg', 0),
        'Wind_Gust_Speed': data['wind'].get('gust', 0),
        'description': data['weather'][0]['description'],
        'country': data['sys'].get('country', 'N/A')
    }

def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data.drop(columns=['RainTomorrow'])
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def predict_future(model, current_value):
    prediction = [current_value]
    for _ in range(5):
        next_val = model.predict(np.array([[prediction[-1]]]))[0]
        prediction.append(next_val)
    return prediction[1:]

def get_wind_direction(deg):
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75),("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75),("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75), ("N", 348.75, 360)
    ]
    return next(point for point, start, end in compass_points if start <= deg < end)

# ========== Streamlit UI ==========

st.title("🌤️ Weather Insight & Rain Predictor")
st.write("Enter a city name to get weather conditions and rain predictions:")

city = st.text_input("City Name")

if city:
    current_weather = get_current_weather(city)
    if current_weather:
        st.subheader(f"📍 {current_weather['city']}, {current_weather['country']}")
        st.write(f"🌡️ Temperature: {current_weather['current_temp']}°C")
        st.write(f"💧 Humidity: {current_weather['humidity']}%")
        st.write(f"🌬️ Wind Speed: {current_weather['wind_speed']} m/s")
        st.write(f"📝 Description: {current_weather['description'].title()}")

        # Load historical data
        data = read_historical_data("weather.csv")
        X, y, le = prepare_data(data)
        rain_model = train_rain_model(X, y)

        wind_dir = get_wind_direction(current_weather['wind_deg'])
        wind_dir_encoded = le.transform([wind_dir])[0] if wind_dir in le.classes_ else -1

        current_df = pd.DataFrame([{
            'MinTemp': current_weather.get('min_temp', 0),
            'MaxTemp': current_weather.get('max_temp', 0),
            'WindGustDir': wind_dir_encoded,
            'WindGustSpeed': current_weather.get('Wind_Gust_Speed', 0),
            'Humidity': current_weather.get('humidity', 0),
            'Pressure': current_weather.get('pressure', 1013),
            'Temp': current_weather.get('current_temp', 0)
        }])

        rain_prediction = rain_model.predict(current_df)[0]
        st.write(f"🌧️ Will it rain tomorrow? **{'Yes' if rain_prediction else 'No'}**")

        # Future prediction
        X_temp, y_temp = prepare_regression_data(data, 'Temp')
        X_hum, y_hum = prepare_regression_data(data, 'Humidity')

        temp_model = train_regression_model(X_temp, y_temp)
        hum_model = train_regression_model(X_hum, y_hum)

        future_temp = predict_future(temp_model, current_weather['current_temp'])
        future_hum = predict_future(hum_model, current_weather['humidity'])

        future_temp = [round(float(val), 2) for val in future_temp]
        future_hum = [round(float(val), 2) for val in future_hum]

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone).replace(minute=0, second=0, microsecond=0)
        future_time = [(now + timedelta(hours=i+1)).strftime("%H:%M") for i in range(5)]

        st.markdown("### 🔮 Forecast (Next 5 Hours)")
        forecast_df = pd.DataFrame({
            "Time": future_time,
            "Predicted Temperature (°C)": future_temp,
            "Predicted Humidity (%)": future_hum
        })
        st.dataframe(forecast_df.set_index("Time"))
