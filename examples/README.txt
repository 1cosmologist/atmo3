
NORMALIZATION PROFILEs FOR JULY 2023 (Q, T AND delta(Q))

Files "mean_q_profiles_2023_07.npz" and "mean_ta_profiles_2023_07.npz" contain 2 arrays: 

mean_profiles: 
	A 2D NumPy array of shape (24, 990)
	→ for q: Hourly mean specific humidity profiles (unit: kg/kg).
	→ for ta: Hourly mean temperature profiles (unit: Kelvin).

	Rows correspond to hours of the day (UTC, from 0 to 23).

	Columns correspond to altitude levels (from 5110 m to 15000 m, every 10 m).

alt_grid: 
	A 1D NumPy array of shape (990,)
	→ Altitude grid in meters.

	

File "delta_q_profiles_2023_07.npz" contain 2 arrays: 

delta_q_map:
	A 2D NumPy array of shape (24, 990)
	→ Hourly humidity fluctuations (e.g., deviation from daily mean).
	
	Rows correspond to hours of the day (UTC, from 0 to 23).

	Columns correspond to altitude levels (from 5110 m to 15000 m, every 10 m).

alt_grid:
	A 1D NumPy array of shape (990,)
	→ Same altitude grid as above (in meters).

