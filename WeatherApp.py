import streamlit as st
from geopy.geocoders import Nominatim
from meteostat import Stations, Point, Daily
from datetime import datetime, date, timedelta

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_locator")

# Streamlit app title
st.title("City Coordinates Finder")

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
    st.subheader(f"Weather Data for {date_check.date()}:")
    st.write(f"Min Temperature: {weather_data['tmin'].iloc[0]}°C")
    st.write(f"Max Temperature: {weather_data['tmax'].iloc[0]}°C")
else:
    st.error("Could not find a weather station near the location.")
