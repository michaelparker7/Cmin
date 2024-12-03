import streamlit as st
from geopy.geocoders import Nominatim

# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_locator")

# Streamlit app title
st.title("City Coordinates Finder")

# Sidebar for user input
city = st.sidebar.text_input("Enter a city name:", "Bethlehem, PA")
selected_date = st.sidebar.date_input("Select a date:", None)
# Fetch coordinates when the user enters a city
if city:
    location = geolocator.geocode(city)
    if location:
        st.subheader(f"Coordinates for {city}:")
        st.write(f"**Latitude:** {location.latitude}")
        st.write(f"**Longitude:** {location.longitude}")
    else:
        st.error(f"Could not find coordinates for '{city}'. Please try again.")


stations = Stations()

start_date = selected_date
end_date = selected_date

ohthatstheone = stations.nearby(location.latitude,location.longitude).fetch().head(1)
ohthatstheone['latitude']
weather_location = Point(ohthatstheone['latitude'],ohthatstheone['longitude'])
weather_location

data = Daily(weather_location, start_date, end_date)
data = data.fetch()

# Print the temperature data
print(f"Temperature on {start_date.date()}:")
print(f"Min: {data['tmin'][0]}°C")
print(f"Max: {data['tmax'][0]}°C")