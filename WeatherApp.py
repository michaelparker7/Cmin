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
    st.sidebar.markdown(
        '<p style="font-size:12px; color:gray;">Works best if formatted as [City, Abbreviated State] (Philadelphia, PA) or [City, Country] (Paris, France)</p>',
        unsafe_allow_html=True,
    )
    city = st.sidebar.text_input("Enter a city name:", "Bethlehem, PA")
    st.sidebar.subheader("When are you walking your dog?")
    
    # Initialize selected_date
    today = datetime.now().date()
    selected_date = st.sidebar.date_input("Select a date:", date(2023, 7, 4))

    # Add "Today" button
    if st.sidebar.button("Today"):
        selected_date = today  # Set selected_date to today
        st.sidebar.success(f"Date set to today: {selected_date}")

    st.sidebar.markdown(
        '<p style="font-size:12px; color:gray;">*If error, refresh page*</p>',
        unsafe_allow_html=True,
    )
    selected_datetime = datetime.combine(selected_date, datetime.min.time())

    
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
    left_col, right_col = st.columns([1, 4])  # Adjust the width ratio as needed
    
    with left_col:
        # Radio buttons for navigation
        section = st.radio(
            "Choose a section:",
            [
                "1: Steady State Solution",
                "2: Air Temperature Over 24 Hours",
                "3: Bird Clear Sky Model",
                "4: Interpolation Function",
                "5: Time Marching Function",
                "6: Recommendations"
            ]
        )
    
    with right_col:
        # Display content based on the selected section
        if section == "1: Steady State Solution":
            st.subheader("1. Steady State Solution:")
            st.write(
                """
                Set up the finite-difference equations and use MATLAB to solve for the steady-state temperature distribution 
                in the concrete and ground soil at midnight on July 4th, assuming that Ts = T∞ at this time point. 
                Plot the temperature distribution for this solution, showing temperature on the x-axis and depth on the y-axis.
                """
            )
            st.image("photos/TaskOne.png", caption="Task 1", use_column_width=False)
            
        elif section == "2: Air Temperature Over 24 Hours":
            st.subheader("2. Air Temperature Over 24 Hours")
            st.write(
                """
                Set up time functions for ambient air temperatures in July for zip code 18015 and your
                chosen second location. Use MATLAB to plot air temperature versus 24-hour time (0:00 to
                24:00 hours). 
                """
            )
            st.image("photos/TaskTwo.png", caption="Task 1", use_column_width=False)
            
        elif section == "3: Bird Clear Sky Model":
            st.subheader("3. Bird Clear Sky Model")
            st.write(
                """
                Use the Bird clear sky model spreadsheet (or another well-justified alternative) to plot
                surface heat flux (i.e. “global horizontal solar flux”, column Y) for the 24 hours of July 4th in
                zip code 18015 and your chosen second location. 
                """
            )
            st.image("photos/TaskThree.png", caption="Task 1", use_column_width=False)
        
        elif section == "4: Interpolation Function":
            st.subheader("4. Interpolation Function")
            st.write(
                """
                Create an interpolation function in MATLAB for the thermophysical properties of air over
                ranges of temperatures that are relevant for this assignment. 
                """
            )
            matlab_code1 = """
            function [rho, k, c, alpha, nu, Pr] = ME321ProjectPt4(T)
               % need density rho, thermal conductivity k, heat capacity c, Fourier # Fo,
               % diffusivity alpha (function of rho, k, c), kinematic viscosity nu, Pr
            
               % Temperature range in degrees celsius
               T_range1 = [-25 -15 -10 -5 0 5 10 15 20 25 30 40 50 60];
               T_range2 = [-100 -50 0 25 50 100];
            
               % Density in kg/m^3 @ atmospheric pressure
               rho_range = [1.422 1.367 1.341 1.316 1.292 1.268 1.246 1.225 1.204 1.184 1.164 1.127 1.093 1.060];
            
               % Thermal conductivity in W/m*K @ atmospheric pressure
               k_range = [22.41 23.20 23.59 23.97 24.36 24.74 25.12 25.50 25.87 26.24 26.62 27.35 28.08 28.80];
               k_range = k_range/1000;
            
               % Specific heat capacity in J/kg*K @ atmospheric pressure
               c_range = [1.005 1.005 1.005 1.005 1.005 1.005 1.005 1.006 1.006 1.006 1.006 1.007 1.007 1.008];
               c_range = c_range*1000;
              
               % Kinematic viscosity in m^2/s @ atmospheric pressure
               nu_range = [11.18 12.01 12.43 12.85 13.28 13.72 14.16 14.61 15.06 15.52 15.98 16.92 17.88 18.86];
               nu_range = nu_range*10^(-6);
              
               % Prandtl number @ atmospheric pressure
               Pr_range = [0.734 0.72 0.711 0.707 0.705 0.701];
            
               % Interpolation of all desired values for any given temperature input
               % using the above ranges of values
               rho = interp1(T_range1, rho_range, T, 'linear');
               k = interp1(T_range1, k_range, T, 'linear');
               c = interp1(T_range1, c_range, T, 'linear');
               alpha = k/(rho*c);
               nu = interp1(T_range1, nu_range, T, 'linear');
               Pr = interp1(T_range2, Pr_range, T, 'linear');
            end
            """
            
            # Display the MATLAB code with syntax highlighting
            st.code(matlab_code, language="matlab")            
        
        elif section == "5: Time Marching Function":
            st.subheader("5. Time Marching Function")
            st.write(
                """
                Set up a time marching function that uses your result from (1) as the initial condition and
                proceeds for 48 hours using the explicit forward-marching finite-difference method.
                Generate one plot showing the temperature distribution in the concrete and ground soil at t
                = 0 (initial steady-state solution), midnight of the second night (t = 24 hours) and noon of the
                second day (t = 36 hours), using the same format as in (1). Generate a second plot showing
                surface temperature, Ts, over the 48-hour period. Complete this step for zip code 18015 and
                your chosen second location. 
                """
            )
            matlab_code2 = """
                %Read in initial conditions from input text files;
                T_init_Vegas = csvread('Vegas_init.txt');
                T_init_Beth = csvread('Beth_init.txt');
                ambient_vegas = csvread('Vegas_amb.txt');
                ambient_Beth = csvread('Beth_amb.txt');
                solar_flux_vegas = csvread('Vegas_flux.txt');
                %set program to solve for location: type either T_init_Vegas or T_init_Beth
                T_init = T_init_Vegas;
                ambdata = ambient_vegas;
                fluxdata = solar_flux_vegas;
                T_dew = 285; %local dew point temp in kelvin
                N = 0; %percent cloud cover
                wind_speed = 10; %average windspeed m/s
                %data and constrats
                Sidewalk_width = 1;
                Sidewalk_length = 1;
                L_re = Sidewalk_width;
                L_gr = (Sidewalk_width^2)/(2*Sidewalk_width+2*Sidewalk_length);
                A_sidewalk = Sidewalk_width*Sidewalk_length;
                %Constant Material Properties
                nodal_spacing = 0.01;
                %Dirt
                k_dirt = 0.52;
                p_dirt = 2050;
                cp_dirt = 1840*1000;
                alpha_dirt = k_dirt/(p_dirt*cp_dirt);
                %Concrete
                k_conc = 1.4;
                p_conc = 2300;
                cp_conc = 880*1000;
                alpha_conc = k_conc/(p_conc*cp_conc);
                emissivity= 0.9;
                solar_abs = 0.65;
                %k_dirt = k_conc;
                %p_dirt = p_conc;
                %cp_dirt = cp_conc;
                %alpha_dirt = k_dirt/(p_dirt*cp_dirt);
                %Node creation
                nodes = ones(1,115);
                nodes(1:13) = 3; %Concrete Nodes
                nodes(13) = 2; %Soil to Concrete Boundary Node
                nodes(1) = 4; %node exposed to thermal radiation and convection
                nodes(length(nodes)) = 5; %sets final layer of model
                %All other nodes at 1
                %Import air properties as function of time
                %Loop Through Time
                t_step = 60; %timestep in seconds
                time_vect = 0:t_step:48*3600;
                %Temp vs time matrix setup
                T = zeros(length(nodes),length(time_vect));
                T(:,1) = T_init; %sets starting node temp
                %Joint node
                Fo_Jdirt = 2*k_dirt * t_step/((p_dirt*cp_dirt+p_conc*cp_conc)*(nodal_spacing^2));
                Fo_Jconc = 2*k_conc * t_step/((p_dirt*cp_dirt+p_conc*cp_conc)*(nodal_spacing^2));
                %Fourier number calcs
                Fo_dirt = alpha_dirt*t_step/(nodal_spacing^2);
                Fo_conc = alpha_conc*t_step/(nodal_spacing^2);
                for i = 1:length(time_vect)
                   %Get air properties at T
                   %Sky Constants:
                   %alpha_air = 22.5*10^-6;
                   %v = 15.89*10^-6; %air kinematic viscosity
                   %Pr = 0.707;
                   %k = 26.3*10^-3; %Air thermal conductivity
                   %T_sky = 60;
                   [rho, k, c, alpha_air, v, Pr] = air_prop(real(T(1,i)));
                   %[T_amb] = ambient_temp(i,ambdata)
                  
                   %[q_raq_flux] = solarflux(i,fluxdata)
                   time_hours(i) = i/60;
                   q_raq_flux = interp1(fluxdata(:,1),fluxdata(:,2),time_hours(i));
                   T_amb = interp1(ambdata(:,1),ambdata(:,2),time_hours(i));
                   %Need a function to calculate convection at top surface (forced+Free)
                   %(outputs q"conv
                   %Free Convection
                   T_film = (T_amb + T(1,i))/2;
                   beta = 1/T_film;
                   Ral = abs((9.8*beta*(T(1,i) - T_amb)*L_gr^3)/(alpha_air*v));
                   Nu_free = 0.52*Ral^(1/5);
                  
                   %Forced Convection
                   Re = wind_speed*L_re/(v);
                   A = 0.037*Re^(4/5)-0.664*Re^(1/2);
                   Nu_forced = (0.037*Re^(4/5)-A)*Pr^(1/3);
                   %Calculate total Convection
                   Nu_total = Nu_forced+ Nu_free;
                   h_bar = Nu_total*k/L_re;
                   q_conv = A_sidewalk*h_bar*(T(1,i)-T_amb);
                  
                   %Need a function to calculate radiation heat transfer outputs q"rad
                   F_cloud = 1 + 0.024*N-0.0035*N^2+0.00028*N^3;
                   e_s = 0.787+0.764*log(T_dew/273)*F_cloud;
                   T_sky = e_s^(.25)*T_amb;
                   q_rad = A_sidewalk*emissivity*(5.67*10^(-8))*(T(1,i)^4-T_sky^4);
                   q_rad_sun = A_sidewalk*solar_abs*q_raq_flux;
                   q_total = q_rad+q_conv+q_rad_sun;
                  
                   for u = 1:length(nodes) %Loop Through Nodes to find temps at t+1
                       if nodes(u) == 1 %Dirt nodes
                           T(u,i+1) = T(u,i)+Fo_dirt*(T(u+1,i)+T(u-1,i)-2*T(u,i));
                       elseif nodes(u) == 2 %dirt concrete interface
                            T(u,i+1) = Fo_Jconc*T(u-1,i)+Fo_Jdirt*T(u+1,i)+ (1-Fo_Jconc-Fo_Jdirt)*T(u,i);
                       elseif nodes(u) == 3 %concrete nodes
                            T(u,i+1) = T(u,i)+Fo_conc*(T(u+1,i)+T(u-1,i)-2*T(u,i));
                       elseif nodes(u) == 4 %surface node
                            T(u,i+1) = T(u,i)+2*Fo_conc*(T(u+1,i)-T(u,i))+2*Fo_conc*q_total*nodal_spacing/(k_conc);
                       elseif nodes(u) == 5 %insulated node
                            T(u,i+1) = T(u,i);%+2*Fo_dirt*(T(u-1,i)-T(u,i));
                       end
                   end
                end
                figure()
                hold on
                title('Temp at Varying Depths')
                xlabel('Time (min)')
                ylabel('Temp (c)')
                plot(1:length(T(1,:)),T(1,:))
                plot(1:length(T(25,:)),T(25,:))
                plot(1:length(T(50,:)),T(50,:))
                plot(1:length(T(75,:)),T(75,:))
                plot(1:length(T(length(nodes),:)),T(length(nodes),:))
                legend('Surface','.25m', '.5m', '.75m', '1.14m')
                hold off
                figure()
                surf(T)
                shading interp
                title('Temp Vs Depth Vs Time')
                xlabel ('time(min)')
                ylabel('Depth (cm)')
                zlabel('Temp (c)')
                figure()
                plot(time_hours,T(1,1:(length(time_hours))))
                title('Concrete Surface Temp')
                xlabel('Time (hours)')
                ylabel('Sidewalk Temp (c)')
                axis([0 48 -inf inf])

            """            
        
        elif section == "6: Recommendations":
            st.subheader("6. Recommendations")
            st.write("This section will provide recommendations.")
            # Add more content here as needed.