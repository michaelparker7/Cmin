import streamlit as st
from geopy.geocoders import Nominatim

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_locator")

# Streamlit app title
st.title("City Coordinates Finder")

# Sidebar for user input
city = st.sidebar.text_input("Enter a city name:", "New York, USA")

# Fetch coordinates when the user enters a city
if city:
    location = geolocator.geocode(city)
    if location:
        st.subheader(f"Coordinates for {city}:")
        st.write(f"**Latitude:** {location.latitude}")
        st.write(f"**Longitude:** {location.longitude}")
    else:
        st.error(f"Could not find coordinates for '{city}'. Please try again.")