<img src="atmo3.png" width="150">

# atmo<sup>3</sup>

Atmo(sphere) cube - a simulation cube for atmospheric emissions in the microwave


### Installation

1. Dependencies: `jax` and `numpy`
2. Clone this git repository: `git clone git@github.com:1cosmologist/atmo3.git`
3. Enter the top level directory: `cd atmo3`
4. Install: `pip install .`


### Usage
The `Atmosphere` is agnostic of the weather-data source: it is initialised with a
dictionary of 1-D profiles (`z` [m], `temperature` [K], `pressure` [Pa],
`specific_humidity` [kg/kg], optional `temperature_std` [K]). Profiles from the
AFGL (1986) reference atmospheres in `data/afgl_data` are provided by
`atmo3.afgl_profiles`:

```python
profiles = a3.afgl_profiles('midlatitude_summer')
atmo = a3.Atmosphere(profiles=profiles, apex_datafile='meteo_apex.csv', ...)
```

### Testing
`python ./examples/test_cube.py`