import streamlit as st
from geopy.geocoders import Nominatim
from meteostat import Stations, Point, Daily
from datetime import datetime, date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from timezonefinder import TimezoneFinder
import pytz
import pandas as pd

st.set_page_config(
    page_title="Will Your Dog's Paws Be Okay?",
    layout="wide",  # Use 'wide' layout for widescreen
    initial_sidebar_state="expanded",  # Optional: Ensure the sidebar is expanded
)


# Initialize the geolocator
geolocator = Nominatim(user_agent="geo_locator")

# Tabs for app structure
tab1, tab2 = st.tabs(["App", "Work"])

with tab1:
    # App tab content
    st.title("Will Your Dog's Paws Be Okay?")

    # Sidebar for user input
    st.sidebar.header("Input Options")
    st.sidebar.subheader("Where are you walking your dog?")
    city = st.sidebar.text_input("Enter a city name:", "Bethlehem, PA")
    st.sidebar.subheader("When are you walking your dog?")
    selected_date = st.sidebar.date_input("Select a date:", date(2023, 7, 4))
    selected_datetime = datetime.combine(selected_date, datetime.min.time())
    today = datetime.now().date()

    
    if city:
        location = geolocator.geocode(city)

    
    # Split layout into two columns
    col1, col2 = st.columns(2)

    
    with col1:
        # Fetch weather station
        stations = Stations()
        if location:
            weather_station = stations.nearby(location.latitude, location.longitude).fetch().head(1)
            if not weather_station.empty:
                weather_location = Point(
                    weather_station['latitude'].values[0], weather_station['longitude'].values[0]
                )

                # Initialize variables
                weather_data = None
                date_check = selected_datetime

                # Adjust for future dates
                if selected_date > today:
                    st.warning(
                        f"The date you selected ({selected_date}) is too far in the future. "
                        "Searching for the closest past date with data..."
                    )
                    # Go back in time
                    while weather_data is None or weather_data.empty:
                        weather_data = Daily(weather_location, date_check, date_check).fetch()
                        if weather_data.empty:
                            date_check -= timedelta(days=1)
                else:
                    # Go forward in time
                    while weather_data is None or weather_data.empty:
                        weather_data = Daily(weather_location, date_check, date_check).fetch()
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

                    st.subheader(f"Weather Data in {city} on {date_check.date()}:")
                    st.write(f"Min Temperature: {min_temp}°C")
                    st.write(f"Max Temperature: {max_temp}°C")

                    # Generate smooth temperature curve
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
                    plt.plot(hours, temperature_curve, marker="o", label="Temperature")
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


    with col2:

        tf = TimezoneFinder()
        if location:
            latitude = location.latitude
            longitude = location.longitude
    
            # Find timezone name
            timezone_name = tf.timezone_at(lat=latitude, lng=longitude)
            if timezone_name:
                timezone = pytz.timezone(timezone_name)
                utc_offset = timezone.utcoffset(datetime.now())
                offset_hours = utc_offset.total_seconds() / 3600
            else:
                st.error("Could not determine the timezone for the selected city.")
    
        # Bird data and graphs
        if 'latitude' in locals() and 'longitude' in locals():
            Latitude = latitude
            Longitude = longitude
            TZ = offset_hours
            Pressure = 1000
            Ozone = 0.3
            H2O = 1.5
            AOD500 = 0.1
            AOD380 = 0.15
            Taua = 0.2758 * AOD380 + 0.35 * AOD500
            Ba = 0.85
            Albedo = 0.2
    
            # disclaimer
            DOY = [selected_date.timetuple().tm_yday]*23
            HR = list(range(1,24))
            ETR = [1367 * (1.00011+ 0.034221 * np.cos(np.radians(2 * np.pi * (d - 1) / 365))+ 0.00128 * np.sin(np.radians(2 * np.pi * (d - 1) / 365))+ 0.000719 * np.cos(np.radians(2 * (2 * np.pi * (d - 1) / 365)))+ 0.000077 * np.sin(np.radians(2 * (2 * np.pi * (d - 1) / 365))))for d in DOY]
            dangle = [6.283185*(d-1)/365 for d in DOY]
            
            bird = pd.DataFrame([DOY,HR,ETR,dangle]).T
            bird.columns = ['DOY','HR','ETR','dangle']
            
            # Intermediate Parameters
            
            bird['DEC'] = (0.006918-0.399912*np.cos(np.radians(bird['dangle']))+0.070257*np.sin(np.radians(bird['dangle']))-0.006758*np.cos(2*np.radians(bird['dangle'])) +0.000907*np.sin(2*np.radians(bird['dangle']))-0.002697*np.cos(3*np.radians(bird['dangle']))+0.00148*np.sin(3*np.radians(bird['dangle'])))*(180/np.pi)
            bird['EQT'] = (0.0000075+0.001868*np.cos(np.radians(bird['dangle']))-0.032077*np.sin(np.radians(bird['dangle']))-0.014615*np.cos(2*np.radians(bird['dangle'])) -0.040849*np.sin(2*np.radians(bird['dangle'])))*(229.18)
            bird['Hour_Angle'] = 15*(bird['HR']-12.5)+(Longitude)-(TZ)*15+bird['EQT']/4
            bird['Zenith_Air'] = np.arccos(np.cos(np.radians(bird['DEC'])) * np.cos(np.radians(Latitude)) * np.cos(np.radians(bird['Hour_Angle'])) +np.sin(np.radians(bird['DEC'])) * np.sin(np.radians(Latitude))) * (180 / np.pi)
            bird['Air_Mass'] = np.where(bird['Zenith_Air'] < 89,1 / (np.cos(np.radians(bird['Zenith_Air'])) +0.15 / (93.885 - bird['Zenith_Air'])**1.25),0)
            
            # Intermediate Results
            
            bird['T_rayleigh'] = np.where(bird['Air_Mass'] > 0,np.exp(-0.0903 * ((Pressure * bird['Air_Mass'] / 1013)**0.84 *(1 + Pressure * bird['Air_Mass'] / 1013 -(Pressure * bird['Air_Mass'] / 1013)**1.01))),0)
            bird['T_ozone'] = np.where(bird['Air_Mass'] > 0, 1 - 0.1611 * (Ozone * bird['Air_Mass']) * (1 + 139.48 * (Ozone * bird['Air_Mass']))**-0.3034 - 0.002715 * (Ozone * bird['Air_Mass']) / (1 + 0.044 * (Ozone * bird['Air_Mass']) + 0.0003 * (Ozone * bird['Air_Mass'])**2), 0)
            bird['T_gases'] = np.where(bird['Air_Mass'] > 0, np.exp(-0.0127 * (bird['Air_Mass'] * Pressure / 1013)**0.26), 0)
            bird['T_water'] = np.where(bird['Air_Mass'] > 0, 1 - 2.4959 * bird['Air_Mass'] * H2O / ((1 + 79.034 * H2O * bird['Air_Mass'])**0.6828 + 6.385 * H2O * bird['Air_Mass']), 0)
            bird['T_aerosol'] = np.where(bird['Air_Mass'] > 0, np.exp(-(Taua**0.873) * (1 + Taua - Taua**0.7088) * bird['Air_Mass']**0.9108), 0)
            bird['TAA'] = np.where(bird['Air_Mass'] > 0, 1 - 0.1 * (1 - bird['Air_Mass'] + bird['Air_Mass']**1.06) * (1 - bird['T_aerosol']), 0)
            bird['rs'] = np.where(bird['Air_Mass'] > 0, 0.0685 + (1 - Ba) * (1 - bird['T_aerosol'] / bird['TAA']), 0)
            bird['Id'] = np.where(bird['Air_Mass'] > 0, 0.9662 * bird['ETR'] * bird['T_aerosol'] * bird['T_water'] * bird['T_gases'] * bird['T_ozone'] * bird['T_rayleigh'], 0)
            bird['IdnH'] = np.where(bird['Zenith_Air'] < 90, bird['Id'] * np.cos(np.radians(bird['Zenith_Air'])), 0)
            bird['Ias'] = np.where(bird['Air_Mass'] > 0,bird['ETR'] * np.cos(np.radians(bird['Zenith_Air'])) * 0.79 * bird['T_ozone'] * bird['T_gases'] * bird['T_water'] * bird['TAA'] * (0.5 * (1 - bird['T_rayleigh']) + Ba * (1 - (bird['T_aerosol'] / bird['TAA']))) / (1 - bird['Air_Mass'] + bird['Air_Mass']**1.02),0)
            bird['GH'] = np.where(bird['Air_Mass'] > 0, (bird['IdnH'] + bird['Ias']) / (1 - Albedo * bird['rs']), 0)
            
            # Decimal Time
            
            bird['Decimal_Time'] = bird['DOY'] + (bird['HR'] - 0.5) / 24
            
            # Model Results
            
            bird['Direct_Beam'] = bird['Id']
            bird['Direct_Hz'] = bird['IdnH']
            bird['Global_Hz'] = bird['GH']
            bird['Dif_Hz'] = bird['Global_Hz'] - bird['Direct_Hz']
    
            if not bird.empty:
                st.subheader("Global Irradiance vs Hour")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")

                plt.figure(figsize=(10, 6))
                plt.plot(bird['HR'], bird['Global_Hz'], marker="o", label="Global Irradiance")
                plt.title("Global Irradiance vs Hour")
                plt.xlabel("Hour of the Day")
                plt.ylabel("Global Irradiance")
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.legend()
                st.pyplot(plt)

with tab2:
    st.title("Work")
    
    # Sub-tabs using a radio button
    sub_tab = st.radio("Choose a section:", ["1: Steady State Solution", 
                                             "2: Air Temperature Over 24 Hours",
                                             "3: Bird Clear Sky Model",
                                             "4: Interpolation Function",
                                             "5: Time Marching Function",
                                             "6: Recommendations",])

    if sub_tab == "1: Steady State Solution":
        st.write("This is content for Sub-tab 1.")
    elif sub_tab == "Sub-tab 2":
        st.write("Welcome to Sub-tab 2!")
        st.write("This is content for Sub-tab 2.")
    elif sub_tab == "Sub-tab 3":
        st.write("Welcome to Sub-tab 3!")
        st.write("This is content for Sub-tab 3.")
