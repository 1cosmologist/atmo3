import jax.numpy as jnp

from . import constants as const
from . import atm_utils as au 
from . import box
from .profiles import validate_profiles

class AtmosphereCalibrator:
    """
    Interpolate standardized atmospheric profiles onto the simulation
    z-grid and, optionally, recalibrate them to match on-site
    measurements.

    On construction the calibrator interpolates the temperature,
    pressure and specific-humidity profiles (see :mod:`profiles`) onto
    the z-grid.  For each of the site statistics provided (mean and
    standard deviation of the temperature and of the column-integrated
    precipitable water vapour, PWV) the corresponding profile or
    fluctuation amplitude is rescaled to match it; statistics that are
    not provided leave the profiles untouched.  The RMS fluctuation
    amplitudes are used as height-dependent scaling profiles by the
    :class:`box.Box` field generator.
    """

    def __init__(self, 
            z_grid: jnp.ndarray = None,
            profiles: dict = None,
            temperature_mean: float = None,
            temperature_std: float = None,
            pwv_mean: float = None,
            pwv_std: float = None
    ) -> None :
        """
        Build the (optionally recalibrated) atmospheric profiles.

        Parameters
        ----------
        z_grid : jnp.ndarray
            Altitude axis of the simulation grid above sea level (m).
        profiles : dict
            Standardized profile dictionary (see :mod:`profiles`) with
            keys ``'z'`` (m), ``'temperature'`` (K), ``'pressure'`` (Pa)
            and ``'specific_humidity'`` (kg kg⁻¹), plus optionally
            ``'temperature_std'`` (K), the RMS temperature fluctuation,
            and ``'specific_humidity_std'`` (kg kg⁻¹), the RMS
            specific-humidity fluctuation.
        temperature_mean : float, optional
            Mean temperature at the site (K).  If given, the
            temperature profile is rescaled so that its value at the
            lowest grid level equals it.
        temperature_std : float, optional
            Standard deviation of the temperature at the site (K).  If
            given, the temperature fluctuation profile is rescaled so
            that its value at the lowest grid level equals it; if
            ``profiles`` has no ``'temperature_std'`` the fluctuation
            is this value at all heights.  Without either, the default
            is 0.1 K at all heights.
        pwv_mean : float, optional
            Mean PWV at the site (mm).  If given, the specific-humidity
            profile is rescaled so that its column-integrated PWV
            equals it.
        pwv_std : float, optional
            Standard deviation of the PWV at the site (mm).  If given,
            :meth:`calibrate_pwv` rescales water-vapour fluctuation
            fields to it; otherwise that method does nothing.

        Attributes
        ----------
        pressure : jnp.ndarray
            Pressure profile on ``z_grid`` (Pa).
        temperature_profile : jnp.ndarray
            Temperature profile on ``z_grid`` (K).
        spec_humidity_profile : jnp.ndarray
            Specific-humidity profile on ``z_grid`` (kg kg⁻¹).
        vir_temperature : jnp.ndarray
            Virtual temperature derived from the temperature and
            specific-humidity profiles (K).
        q2rho_h2o : jnp.ndarray
            Conversion factor from specific humidity to water-vapour
            mass density: ``p / (R_dry · T_virtual)`` (kg m⁻³ / (kg kg⁻¹)).
        temp_fluctuation_profile : jnp.ndarray
            Height-dependent RMS temperature-fluctuation amplitude (K).
        spec_humidity_fluctuation_profile : jnp.ndarray
            Height-dependent RMS specific-humidity fluctuation amplitude
            (kg kg⁻¹). Defaults to 0.1 % of the calibrated mean profile.
        temperature_mean, temperature_std, pwv_mean, pwv_std : float or None
            The site statistics as provided.
        """
        profiles = validate_profiles(profiles)
        z_grid = jnp.asarray(z_grid)

        for name, value, positive in (('temperature_mean', temperature_mean, True),
                                      ('temperature_std', temperature_std, False),
                                      ('pwv_mean', pwv_mean, True),
                                      ('pwv_std', pwv_std, False)):
            if value is not None and not (value > 0 if positive else value >= 0):
                raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'}, got {value}.")

        self.temperature_mean = temperature_mean
        self.temperature_std  = temperature_std
        self.pwv_mean         = pwv_mean
        self.pwv_std          = pwv_std

        # Pressure falls roughly exponentially with height, so interpolate its logarithm.
        self.pressure = jnp.exp(jnp.interp(z_grid, profiles['z'], jnp.log(profiles['pressure']), left='extrapolate', right='extrapolate'))
        self.temperature_profile = jnp.interp(z_grid, profiles['z'], profiles['temperature'], left='extrapolate', right='extrapolate')
        self.spec_humidity_profile = jnp.clip(
            jnp.interp(z_grid, profiles['z'], profiles['specific_humidity'], left='extrapolate', right='extrapolate'),
            0.0
        )

        if 'temperature_std' in profiles:
            self.temp_fluctuation_profile = jnp.interp(z_grid, profiles['z'], profiles['temperature_std'], left='extrapolate', right='extrapolate')
        else:
            self.temp_fluctuation_profile = 0.1 * jnp.ones_like(self.temperature_profile)

        self._mean_temperature_normalization = 1.0
        self._temperature_fluctuation_norm = 1.0
        self._mean_spec_humidity_normalization = 1.0

        if temperature_mean is not None:
            self._mean_temperature_normalization = temperature_mean / self.temperature_profile[0]
            self.temperature_profile = self.temperature_profile * self._mean_temperature_normalization

        if temperature_std is not None:
            if self.temp_fluctuation_profile[0] <= 0:
                raise ValueError("Cannot rescale the temperature fluctuations: the profile is zero at the lowest grid level.")
            self._temperature_fluctuation_norm = temperature_std / self.temp_fluctuation_profile[0]
            self.temp_fluctuation_profile = self.temp_fluctuation_profile * self._temperature_fluctuation_norm

        self._update_water_vapor_conversion()

        if pwv_mean is not None:
            # The virtual temperature, hence the density conversion, depends on q: iterate to convergence.
            for _ in range(20):
                factor = pwv_mean / jnp.trapezoid(self.q2rho_h2o * self.spec_humidity_profile, x=z_grid)
                self._mean_spec_humidity_normalization *= factor
                self.spec_humidity_profile = self.spec_humidity_profile * factor
                self._update_water_vapor_conversion()
                if abs(factor - 1.0) < 1e-12:
                    break

        if 'specific_humidity_std' in profiles:
            self.spec_humidity_fluctuation_profile = jnp.interp(z_grid, profiles['z'], profiles['specific_humidity_std'], left='extrapolate', right='extrapolate')
        else:
            self.spec_humidity_fluctuation_profile = 1e-3 * self.spec_humidity_profile

    def _update_water_vapor_conversion(self) -> None:
        self.vir_temperature = au.virtual_temperature(self.temperature_profile, self.spec_humidity_profile)
        self.q2rho_h2o = self.pressure / (const.R_dry_air * self.vir_temperature)
        
    def calibrate_pwv(
        self,
        z_axis: jnp.ndarray,
        water_box: type[box.Box] = None
    ) -> None :
        """
        Rescale a water-vapour fluctuation field to match the site PWV
        standard deviation.

        The column-integrated PWV is computed for each horizontal pixel
        of ``water_box.field`` by integrating along the vertical axis.
        The resulting sky-plane PWV map has a standard deviation
        ``sigma_pwv``; the field is then multiplied by
        ``pwv_std / sigma_pwv`` so that the simulated PWV fluctuations
        have the amplitude measured at the site.  If ``pwv_std`` was not
        provided, the field is left unchanged.

        The normalisation factor is stored as
        ``self._spec_hum_fluctuation_norm`` for later inspection.

        Parameters
        ----------
        z_axis : jnp.ndarray
            1-D altitude array corresponding to the third axis of
            ``water_box.field`` (m).  Passed as the ``x`` argument to
            ``jnp.trapezoid``.
        water_box : box.Box
            Water-vapour :class:`box.Box` instance whose ``field``
            attribute (shape ``[Nx, Ny, Nz]``) is rescaled in-place.
        """
        if self.pwv_std is None:
            return

        pwv_plane = jnp.trapezoid(water_box.field, x=z_axis, axis=2)
        sigma_pwv = jnp.std(pwv_plane)
        
        self._spec_hum_fluctuation_norm = self.pwv_std / sigma_pwv 
        
        water_box.field = water_box.field * self._spec_hum_fluctuation_norm
