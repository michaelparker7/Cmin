import streamlit as st
from geopy.geocoders import Nominatim
from meteostat import Stations, Point, Daily
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_locator")

# Streamlit app title
st.title("City Coordinates Finder and Temperature Curve")

# Sidebar for user input
city = st.sidebar.text_input("Enter a city name:", "Bethlehem, PA")
selected_date = st.sidebar.date_input("Select a date:", date(2023, 7, 4))
selected_datetime = datetime.combine(selected_date, datetime.min.time())

# Fetch coordinates when the user enters a city
if city:
    location = geolocator.geocode(city)
    if location:
        st.subheader(f"Coordinates for {city}:")
        st.write(f"**Latitude:** {location.latitude}")
        st.write(f"**Longitude:** {location.longitude}")
    else:
        st.error(f"Could not find coordinates for '{city}'. Please try again.")

# Fetch weather station
stations = Stations()
weather_station = stations.nearby(location.latitude, location.longitude).fetch().head(1)

if not weather_station.empty:
    weather_location = Point(weather_station['latitude'].values[0], weather_station['longitude'].values[0])

    # Initialize variables
    weather_data = None
    date_check = selected_datetime

    # Check for available weather data
    while weather_data is None or weather_data.empty:
        weather_data = Daily(weather_location, date_check, date_check).fetch()

        # If no data, move to the next day
        if weather_data.empty:
            date_check += timedelta(days=1)

    # Inform the user if the date had to change
    if date_check.date() != selected_date:
        st.warning(
            f"No weather data available for the day you selected ({selected_date}). "
            f"The closest available date is {date_check.date()}."
        )

    # Display weather data
    if not weather_data.empty:
        min_temp = weather_data['tmin'].iloc[0]
        max_temp = weather_data['tmax'].iloc[0]

        st.subheader(f"Weather Data for {date_check.date()}:")
        st.write(f"Min Temperature: {min_temp}°C")
        st.write(f"Max Temperature: {max_temp}°C")

        # Function to generate smooth temperature curve
        def generate_smooth_temperature_curve(min_temp, max_temp):
            key_hours = np.array([0, 5, 9, 15, 20, 23])  # Key points
            key_temps = np.array([
                min_temp, min_temp, (min_temp + max_temp) / 2, max_temp,
                (min_temp + max_temp) / 2, min_temp
            ])
            spline = CubicSpline(key_hours, key_temps, bc_type='natural')
            hours = np.arange(24)
            temperatures = spline(hours)
            return hours, temperatures

        # Generate and plot temperature curve
        hours, temperature_curve = generate_smooth_temperature_curve(min_temp, max_temp)

        plt.figure(figsize=(10, 6))
        plt.plot(hours, temperature_curve, marker="o", label="Estimated Temperature")
        plt.title(f"Temperature Curve for {city} on {date_check.date()}")
        plt.xlabel("Hour of the Day")
        plt.ylabel("Temperature (°C)")
        plt.xticks(hours)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)
else:
    st.error("Could not find a weather station near the location.")
