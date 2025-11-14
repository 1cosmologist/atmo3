import scipy.constants as con 

g = con.g # m s-2

molar_mass_dry_air = 28.9647e-3 # kg mol-1
R_dry_air = con.gas_constant / molar_mass_dry_air # J mol-1 K-1 / (kg mol-1) = J kg-1 K-1 = m-2 s-2 K-1

molar_mass_water_vapor = 18.01528e-3 # kg mol-1
R_water_vapor = con.gas_constant / molar_mass_water_vapor # J mol-1 K-1 / (kg mol-1) = J kg-1 K-1 = m-2 s-2 K-1

pressure_at_sea_level  = 101325.0  # Sea level standard atmospheric pressure in Pa
temperature_at_sea_level = 288.15    # Sea level standard temperature in K
temperature_lapse_rate = 0.0065     # Temperature lapse rate in K/m


