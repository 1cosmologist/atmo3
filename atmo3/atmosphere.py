from . import box
from . import super_grid
from . import calibration 
from . import grid_utils as gutl
from . import constants  as const
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime, timezone
import gc

class Atmosphere:
    """
    A 3D atmospheric simulation box that models turbulent fluctuations
    of physical fields (e.g., temperature, water vapor density) over a
    rectangular grid.

    The atmosphere is initialized from ERA5 reanalysis profiles and
    optionally calibrated against on-site meteorological measurements
    (APEX weather station data).  Each atmospheric component is stored
    as a :class:`box.Box` instance whose fluctuations are drawn from a
    user-supplied power spectrum and rescaled by a height-dependent
    profile derived from ERA5 data.
    """

    def __init__(self, 
                 nside_grid: list = [256, 256, 128], 
                 box_length_in_m: list = [20000.0, 20000.0, 10000.0],
                 site_altitude: float = 0.0,
                 site_coordinates: list = [0.0, 0.0],
                 time_utc: type[datetime] = datetime(2023, 9, 1, 0, 0, tzinfo=timezone.utc),
                 geopotential_file_era5: str = None,
                 temperature_file_era5: str = None,
                 spec_humidity_file_era5: str = None,
                 apex_datafile: str = None
        ) -> None:
        
        """
        Initialize the Atmosphere simulation box.

        Parameters
        ----------
        nside_grid : list of int, optional
            Number of grid cells along each axis [Nx, Ny, Nz].
            Defaults to [256, 256, 128].
        box_length_in_m : list of float, optional
            Physical size of the simulation box in metres along each
            axis [Lx, Ly, Lz].  Defaults to [20000.0, 20000.0, 10000.0].
        site_altitude : float, optional
            Altitude of the observing site above sea level in metres.
            Defaults to 0.0.
        site_coordinates : list of float, optional
            Geographic coordinates of the site as [longitude, latitude]
            in degrees. Defaults to [0.0, 0.0].
        time_utc : datetime, optional
            UTC timestamp of the simulation epoch.  Used to select the
            matching ERA5 profile and APEX weather data.
            Defaults to 2023-09-01 00:00 UTC.
        geopotential_file_era5 : str, optional
            Path to an ERA5 geopotential NetCDF file used to build the
            vertical coordinate of the super-grid.
        temperature_file_era5 : str, optional
            Path to an ERA5 temperature NetCDF file used for mean and
            fluctuation profiles.
        spec_humidity_file_era5 : str, optional
            Path to an ERA5 specific-humidity NetCDF file used for mean
            and fluctuation profiles.
        apex_datafile : str, optional
            Path to a CSV file containing APEX weather-station
            measurements (PWV, temperature, humidity, wind) used to
            calibrate the simulated profiles.

        Attributes
        ----------
        N : jnp.ndarray
            Grid dimensions [Nx, Ny, Nz].
        Lbox : jnp.ndarray
            Box dimensions in metres [Lx, Ly, Lz].
        site_altitude : float
            Site altitude in metres.
        session_time : tuple
            UTC timestamp of the simulation (stored as a one-element
            tuple due to trailing comma assignment).
        site_coordinates : jnp.ndarray
            Site [longitude, latitude] in degrees.
        grid_wsp : gutl.GridWorkspace
            Grid workspace providing coordinate axes, wavenumber grids,
            and interpolation helpers.
        super_grid : super_grid.SuperGrid
            Vertical super-grid built from ERA5 geopotential data.
        atm_calibrator : calibration.AtmosphereCalibrator
            Calibrator that normalises ERA5 profiles to match on-site
            observations.
        component_names : list of str
            Ordered list of registered component names.
        components : dict
            Mapping from component name to its :class:`box.Box` instance.
        component_mean : dict
            Mapping from component name to its mean-profile dictionary
            ``{'h': heights, 'f': mean_values}``.
        """
        self.N                = jnp.array(nside_grid)
        self.Lbox             = jnp.array(box_length_in_m)
        self.site_altitude    = site_altitude
        self.session_time     = time_utc,
        self.site_coordinates = jnp.array(site_coordinates)
        self.grid_wsp         = gutl.GridWorkspace(
                                                N=self.N, 
                                                Lbox=self.Lbox, 
                                                site_altitude=self.site_altitude
                                                )
        
        self.super_grid       = super_grid.SuperGrid(
                                                geopotential_file=geopotential_file_era5,
                                                z_max=self.Lbox[2],
                                                Nz=self.N[2],
                                                time_utc=self.session_time,
                                                site_lonlat=self.site_coordinates,
                                                site_altitude=self.site_altitude
                                                )
        
        self.atm_calibrator   = calibration.AtmosphereCalibrator(
                                                super_grid=self.super_grid,
                                                temperature_file=temperature_file_era5,
                                                sp_humidity_file=spec_humidity_file_era5,
                                                apexdatafile=apex_datafile,
                                                )
        
        self.component_names  = []
        self.components       = {}
        self.component_mean   = {}

    def _add_component(
        self,
        field_name: str,
        field_unit: str,
        pspec: dict,
        zscale: dict,
        seed: int,
        mean: dict = None,
        nsub: int = 1024**3
    ) -> None:
        
        """
        Register a new turbulent field component in the atmosphere.

        Instantiates a :class:`box.Box` for the requested field and
        records its mean profile so that callers can later add the mean
        to the generated fluctuation field.

        Parameters
        ----------
        field_name : str
            Human-readable name used as the dictionary key
            (e.g. ``'temperature'``, ``'water vapor'``).
        field_unit : str
            Physical unit of the field (e.g. ``'K'``, ``'kg / m^3'``).
        pspec : dict
            Isotropic power spectrum with keys:

            - ``'k'``    - wavenumber array (rad m⁻¹)
            - ``'pofk'`` - power at each wavenumber
        zscale : dict
            Height-dependent RMS scaling profile with keys:

            - ``'h'`` - height array (m)
            - ``'f'`` - amplitude scaling at each height
        seed : int
            Master random seed passed to the parallel RNG stream.
        mean : dict, optional
            Mean vertical profile with keys:

            - ``'h'`` - height array (m)
            - ``'f'`` - mean field value at each height

            Defaults to ``None`` (zero mean).
        nsub : int, optional
            Sub-stream length for the parallel RNG.  Defaults to
            ``1024**3``.
        """
        
        self.component_names.append(field_name)
        self.components[field_name] = box.Box(
                                        grid_wsp=self.grid_wsp,
                                        field_name=field_name,
                                        field_unit=field_unit,
                                        spectrum=pspec,
                                        zscaling=zscale,
                                        seed=seed,
                                        nsub=nsub,                               
                                    )
        self.component_mean[field_name] = mean

    
    def add_temperature(self,
                        power_spec: dict, 
                        seed: int = 13579, 
                        ) -> None :
        """
        Add a turbulent temperature component to the atmosphere.

        The height-dependent RMS amplitude and mean profile are taken
        from the ERA5-calibrated :attr:`atm_calibrator`:

        - Fluctuation scaling: ``atm_calibrator.temp_fluctuation_profile``
        - Mean profile:        ``atm_calibrator.temperature_profile``

        Parameters
        ----------
        power_spec : dict
            Isotropic power spectrum with keys:

            - ``'k'``    - wavenumber array (rad m⁻¹)
            - ``'pofk'`` - power at each wavenumber
        seed : int, optional
            Master random seed for the temperature RNG stream.
            Defaults to 13579.
        """
        self._add_component(
            field_name='temperature',
            field_unit='K',
            pspec=power_spec,
            zscale={'h': self.super_grid.z, 'f':self.atm_calibrator.temp_fluctuation_profile},
            seed=seed,
            mean= {'h': self.super_grid.z, 'f': self.atm_calibrator.temperature_profile},
        )
        
    def add_watervapor(self,
                       power_spec:dict,
                       seed: int = 24680,
                       ) -> None :
        """
        Add a turbulent water-vapor density component to the atmosphere.

        The ERA5 specific-humidity profiles are converted to water-vapor
        mass density (kg m⁻³) via the factor
        ``atm_calibrator.q2rho_h2o = p / (R_dry · T_virtual)``:

        - Fluctuation scaling: ``atm_calibrator.spec_humidity_fluctuation_profile * q2rho_h2o``
        - Mean profile:        ``atm_calibrator.spec_humidity_profile * q2rho_h2o``

        Parameters
        ----------
        power_spec : dict
            Isotropic power spectrum with keys:

            - ``'k'``    - wavenumber array (rad m⁻¹)
            - ``'pofk'`` - power at each wavenumber
        seed : int, optional
            Master random seed for the water-vapor RNG stream.
            Defaults to 24680.
        """
        self._add_component(
            field_name='water vapor',
            field_unit='kg / m^3',
            pspec=power_spec,
            zscale={'h': self.super_grid.z, 'f':self.atm_calibrator.spec_humidity_fluctuation_profile*self.atm_calibrator.q2rho_h2o},
            seed=seed,
            mean= {'h': self.super_grid.z, 'f': self.atm_calibrator.spec_humidity_profile*self.atm_calibrator.q2rho_h2o},
        )




    def derive_cloud_mask(self, ccfile: str):
        """
        Derives a binary 3D cloud mask (0 or 1) based on the generated 
        temperature and water vapor fields, calibrated against ERA5 cloud cover.
        If temperature fluctuations are not generated, it uses the mean temperature profile.
        
        Parameters
        ----------
        ccfile : str
            Path to the ERA5 cloud cover (.nc) file.
            
        Returns
        -------
        jnp.ndarray
            A 3D binary array of shape [Nx, Ny, Nz] representing cloud presence.
        """

        import jax
        import jax.numpy as jnp
        import warnings
        from atmo3 import atm_utils as au

        # 1. Verify water vapor exists (this is mandatory)
        if 'water vapor' not in self.components:
            raise ValueError("The 'water vapor' component must be generated first.")

        # 2. Interpolate the ERA5 cloud cover ratio to the simulation's altitude grid
        cc_interp = self.super_grid.era5_interp2site(ccfile)

        # 3. Reconstruct the TOTAL 3D Temperature field
        if 'temperature' not in self.components:
            print("Warning: 'temperature' component not found. Using the mean ERA5 temperature profile (no fluctuations) to derive the cloud mask.")
            # Fallback to the mean profile already stored in the calibrator
            t_total_3d = self.atm_calibrator.temperature_profile.reshape(1, 1, -1)
        else:
            t_mean_1d = self.component_mean['temperature']['f']
            t_fluct_3d = self.components['temperature'].field
            t_total_3d = t_fluct_3d + t_mean_1d.reshape(1, 1, -1)

        # 4. Reconstruct the TOTAL 3D Water Vapor Density field (Mean + Fluctuations)
        q_mean_1d = self.component_mean['water vapor']['f']
        q_fluct_3d = self.components['water vapor'].field
        q_total_3d = q_fluct_3d + q_mean_1d.reshape(1, 1, -1)

        # 5. Calculate true 3D Relative Humidity
        rh_total_3d = au.water_vapor_density_to_rel_humidity(rho_wv=q_total_3d, T=t_total_3d) * 100.0

        # 6. Ultra-optimized JAX quantile thresholding
        @jax.jit
        def _compute_mask(rh_cube, cc_prof):
            nz = rh_cube.shape[2]
            
            # Flatten spatial dimensions for map-reduce efficiency: (nz, nx*ny)
            rh_flat = jnp.transpose(rh_cube, (2, 0, 1)).reshape(nz, -1)
            
            # Calculate target quantiles: if CC is 0.2, we want the top 20% (quantile 0.8)
            q_targets = jnp.clip(1.0 - cc_prof, 0.0, 1.0)
            
            # Evaluate the exact threshold for all altitude layers simultaneously
            layer_thresholds = jax.vmap(jnp.quantile)(rh_flat, q_targets)
            
            # Broadcast back to 3D shape (1, 1, nz)
            thresholds_3d = layer_thresholds.reshape(1, 1, nz)
            cc_3d = cc_prof.reshape(1, 1, nz)
            
            # Create binary mask: 1 if RH >= threshold AND ERA5 cc > 0, else 0
            cloud_cube = jnp.where(cc_3d > 0.0, rh_cube >= thresholds_3d, 0)
            
            return cloud_cube.astype(jnp.int8) # int8 is sufficient for binary masks

        # 7. Generate and store the mask
        self.cloud_mask = _compute_mask(rh_total_3d, cc_interp)
        
        return self.cloud_mask

    def derive_liquid_ice_cubes(self, ccfile: str, ciwcfile: str = None, clwcfile: str = None):
        """
        Generates 3D cubes of Cloud Ice Water Content (CIWC) and Cloud Liquid Water Content (CLWC).
        The values are distributed only into voxels where the cloud mask is active.
        
        Parameters
        ----------
        ccfile : str
            Path to the ERA5 cloud cover (.nc) file.
        ciwcfile : str, optional
            Path to the ERA5 cloud ice water content (.nc) file.
        clwcfile : str, optional
            Path to the ERA5 cloud liquid water content (.nc) file.
            
        Returns
        -------
        dict
            A dictionary containing the generated 3D JAX arrays: {'ice': ice_cube, 'liquid': liquid_cube}
        """

        # 1. Get the 3D binary cloud mask (this inherently validates T and q as well)
        # It will either generate it or use the cached one if already run.
        if not hasattr(self, 'cloud_mask'):
            self.derive_cloud_mask(ccfile)
            
        cloud_mask = self.cloud_mask

        # 2. Get the 1D cloud cover ratio for the denominator
        cc_interp = self.super_grid.era5_interp2site(ccfile)
        
        generated_cubes = {}

        # 3. Process Cloud Ice Water Content
        if ciwcfile is not None:
            ciwc_interp = self.super_grid.era5_interp2site(ciwcfile)
            
            # Check if there is actually any ice in this profile
            if jnp.all(ciwc_interp == 0.0):
                print("Notice: No cloud ice water found in the profile. Ice cube will not be generated.")
            else:
                # Calculate in-cloud concentration: ciwc / cc
                # Use jnp.where to prevent divide-by-zero errors in clear-sky layers
                ice_ratio_1d = jnp.where(cc_interp > 0.0, ciwc_interp / cc_interp, 0.0)
                
                # Map 1D profile across the active 3D cloud voxels
                self.ice_cube = cloud_mask * ice_ratio_1d.reshape(1, 1, -1)
                generated_cubes['ice'] = self.ice_cube

        # 4. Process Cloud Liquid Water Content
        if clwcfile is not None:
            clwc_interp = self.super_grid.era5_interp2site(clwcfile)
            
            # Check if there is actually any liquid in this profile
            if jnp.all(clwc_interp == 0.0):
                print("Notice: No cloud liquid water found in the profile. Liquid cube will not be generated.")
            else:
                # Calculate in-cloud concentration: clwc / cc
                # Use jnp.where to prevent divide-by-zero errors in clear-sky layers
                liquid_ratio_1d = jnp.where(cc_interp > 0.0, clwc_interp / cc_interp, 0.0)
                
                # Map 1D profile across the active 3D cloud voxels
                self.liquid_cube = cloud_mask * liquid_ratio_1d.reshape(1, 1, -1)
                generated_cubes['liquid'] = self.liquid_cube

        # 5. Final check
        if not generated_cubes:
            print("Warning: Neither ice nor liquid cubes were generated. (Files were missing or empty).")

        return generated_cubes


    def generate_realization(
        self,
        time_step: int = 0,
        component_name: str = None
    ) -> None:
        
        """
        Generate a random-field realization for one or all components.

        This calls :meth:`box.Box.generate_field_realization` for the
        requested component(s).  If a water-vapor component is
        (re-)generated, :meth:`calibration.AtmosphereCalibrator.calibrate_pwv`
        is called automatically to rescale the field so that its
        column-integrated PWV matches the APEX measurement.

        Parameters
        ----------
        time_step : int, optional
            Index that seeds the RNG sub-stream, allowing deterministic
            time-ordered realizations.  Defaults to 0.
        component_name : str, optional
            Name of the specific component to generate
            (e.g. ``'temperature'``, ``'water vapor'``).  If ``None``
            or not found in :attr:`component_names`, all registered
            components are generated.  Defaults to ``None``.
        """
        if component_name in self.component_names:
            self.components[component_name].generate_field_fluctuations(time_step=int(time_step))
            if component_name == 'water vapor': self.atm_calibrator.calibrate_pwv(self.grid_wsp.grid_axis(axis=2, altitude_axis=True), self.components[component_name])
        else:
            for component in self.components.values():
                component.generate_field_fluctuations(time_step=int(time_step))
                if component.field_name == 'water vapor': self.atm_calibrator.calibrate_pwv(self.grid_wsp.grid_axis(axis=2, altitude_axis=True), component)
                