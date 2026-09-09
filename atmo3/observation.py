"""Line-of-sight construction, sampling, and integration utilities."""

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from .grid_utils import GridWorkspace, NestedGridWorkspace
from .super_grid import SuperGrid
from . import obs_utils

class Observer:
    """Construct and sample telescope lines of sight through atmospheric grids.

    A regular workspace produces one LOS array. A nested workspace produces
    one LOS array per refinement level, with physical path spacing inherited
    from that level's vertical cell size. Missing-scale fields can therefore
    remain on separate devices and be reduced independently.
    """
    
    def __init__(
        self,
        grid_wsp: GridWorkspace | NestedGridWorkspace = None,
        super_grid: type[SuperGrid] = None,
        northwind_era5_file: str = None,
        eastwind_era5_file: str = None,
        boresight: jnp.ndarray = None,       # in grid coordinates
        passband: dict = None, 
        fwhm_arcmin: float = None
    ):
        """Initialize an observer and validate its boresight.

        Parameters
        ----------
        grid_wsp : GridWorkspace or NestedGridWorkspace
            Geometry of the atmospheric realization.
        super_grid : SuperGrid, optional
            Atmospheric profile grid used to interpolate wind profiles onto
            nested altitude axes.
        northwind_era5_file, eastwind_era5_file : str, optional
            ERA5 wind files. Both are required to enable frozen-flow shifts.
        boresight : jax.Array, shape (3,)
            Telescope position in global grid coordinates. Its x/y values must
            lie in the central half of the coarse box.
        passband : dict
            Instrument passband metadata containing ``"freq_GHz"`` and
            ``"g_nu"`` arrays.
        fwhm_arcmin : float
            Beam full width at half maximum in arcminutes.

        Attributes
        ----------
        axes : tuple
            Three coordinate axes for a regular grid, or one three-axis tuple
            per nested level.
        los_obj : jax.Array or tuple of jax.Array
            Populated by :meth:`compute_los_for_scan`.
        path_element_lengths : jax.Array or tuple of jax.Array
            Per-sample physical path increments matching ``los_obj``.
        """
        self.grid_wsp = grid_wsp
        self.super_grid = super_grid
        
        self.north_wind = None; self.east_wind = None
        
        if (not northwind_era5_file == None) and (not eastwind_era5_file == None):
            self.north_wind = self.super_grid.era5_interp2site(northwind_era5_file)
            self.east_wind  = self.super_grid.era5_interp2site(eastwind_era5_file)
            
        
        if not isinstance(boresight, jnp.ndarray):
            raise TypeError("Boresight has to provided.")
        
        coarse_grid = (
            self.grid_wsp.coarse
            if isinstance(self.grid_wsp, NestedGridWorkspace)
            else self.grid_wsp
        )
        lower = coarse_grid.origin + coarse_grid.Lbox / 4
        upper = coarse_grid.origin + 3 * coarse_grid.Lbox / 4

        if not (lower[0] < boresight[0] < upper[0]):
            raise ValueError("Boresight x-coord is incorrectly set.")    
        
        if not (lower[1] < boresight[1] < upper[1]):
            raise ValueError("Boresight y-coord is incorrectly set.")  
        
        self.boresight = boresight 
        
        if not isinstance(passband, dict):
            raise ValueError("Passband required! Format: dict{'freq_GHz', 'g_nu'}.")

        self.passband = passband
        ### TODO: Normalization of the passband is TBD. Assume it is normalized and in units of K_RJ
        
        if not isinstance(fwhm_arcmin, float):
            raise ValueError("FWHM in arcmin required!")        
        
        self.fwhm_arcmin = fwhm_arcmin
        
        if isinstance(self.grid_wsp, NestedGridWorkspace):
            self.axes = tuple(level.axes() for level in self.grid_wsp.levels)
        else:
            self.axes = self.grid_wsp.axes()
        
        
        ### TODO: Calculate max time before refreshing simulation
        
    def compute_los_for_scan(
        self,
        timelist: list,
        azimuth_deg: list,
        elevation_deg: list
    ):
        """Compute LOS coordinates for a sequence of telescope pointings.
        
        Parameters
        ----------
        timelist : list
            List of datetime-like objects representing the observation times.
        azimuth_deg : list
            List of azimuth angles in degrees, length matches timelist representing n_scans.
        elevation_deg : list
            List of elevation angles in degrees, length matches timelist representing n_scans.
            
        Returns
        -------
        None
            For a regular grid, sets ``los_obj`` to shape
            ``(n_scans, Nz, 5)``. For nested geometry, sets ``los_levels`` and
            ``los_obj`` to a tuple containing one such array per level. The
            final coordinate contains ``[x, y, altitude, radius, valid]``.
            ``path_element_lengths`` is created with the same regular/nested
            structure.

        Notes
        -----
        Every refinement uses its own altitude samples. Fine missing-scale
        fields therefore contribute only near the telescope and are evaluated
        with shorter physical path elements.
        """
        timearray = np.array(timelist, dtype='datetime64[ns]')
        delta_t_in_s   = (timearray - timearray[0]) / np.timedelta64(1, 's')
        
        azimuth   = jnp.deg2rad(jnp.array(azimuth_deg))
        elevation = jnp.deg2rad(jnp.array(elevation_deg))
        
        x = jnp.cos(elevation) * jnp.cos(azimuth)
        y = jnp.cos(elevation) * jnp.sin(azimuth)
        z = jnp.sin(elevation)
        
        pos_vec = jnp.array([x, y, z]) / jnp.sqrt(x**2. + y**2. + z**2.)
        def los_for_grid(grid):
            """Build all scan LOS coordinates for one regular grid level."""
            alt_arr = grid.grid_axis(axis=2)
            north_wind = self.north_wind
            east_wind = self.east_wind
            if north_wind is not None and isinstance(
                self.grid_wsp, NestedGridWorkspace
            ):
                north_wind = jnp.interp(alt_arr, self.super_grid.z, north_wind)
                east_wind = jnp.interp(alt_arr, self.super_grid.z, east_wind)
            return jax.vmap(
                lambda uv, dt: obs_utils.los_points_coords_radius(
                    grid.site_altitude,
                    grid.Lbox,
                    alt_arr,
                    uv,
                    self.boresight,
                    north_wind=north_wind,
                    east_wind=east_wind,
                    delta_t=dt,
                    max_radius=True,
                ),
                in_axes=(1, 0),
            )(pos_vec, jnp.asarray(delta_t_in_s))

        if isinstance(self.grid_wsp, NestedGridWorkspace):
            self.los_levels = tuple(
                los_for_grid(level) for level in self.grid_wsp.levels
            )
            self.los_obj = self.los_levels
            self.path_element_lengths = tuple(
                jnp.diff(
                    los[:, :, 3],
                    axis=1,
                    prepend=jnp.zeros((los.shape[0], 1)),
                )
                for los in self.los_levels
            )
        else:
            self.los_obj = los_for_grid(self.grid_wsp)
            self.path_element_lengths = jnp.diff(
                self.los_obj[:, :, 3],
                axis=1,
                prepend=jnp.zeros((self.los_obj.shape[0], 1)),
            )
        
    def scan_component(
        self,
        component_field,
    ):
        """Interpolate a field or independent refinement fields along LOS.

        Parameters
        ----------
        component_field : jax.Array, sequence of jax.Array, or NestedBox
            One regular field, or one missing-scale field per nested level.
            A ``NestedBox``-like object is accepted through its ``fields``
            attribute.

        Returns
        -------
        jax.Array or tuple of jax.Array
            Regular geometry returns shape ``(n_scans, Nz)``. Nested geometry
            returns one such array per level. Samples outside an individual
            level's physical bounds are zero. Nested results remain on the
            devices holding their corresponding three-dimensional fields.
        """
        if not isinstance(self.grid_wsp, NestedGridWorkspace):
            shape = self.los_obj.shape
            points = self.los_obj[:, :, 0:3].reshape(shape[0] * shape[1], 3)
            interpol = jsp.interpolate.RegularGridInterpolator(
                self.axes, component_field, fill_value=0.0, method="linear"
            )
            return interpol(points).reshape(shape[0], shape[1])

        fields = component_field.fields if hasattr(component_field, "fields") else component_field
        if len(fields) != self.grid_wsp.nlevels:
            raise ValueError("One component field is required for every refinement level.")

        samples = []
        for level, axes, field, los in zip(
            self.grid_wsp.levels, self.axes, fields, self.los_levels
        ):
            shape = los.shape
            points = los[:, :, 0:3].reshape(shape[0] * shape[1], 3)
            device = next(iter(field.devices()))
            points = jax.device_put(points, device)
            device_axes = tuple(jax.device_put(axis, device) for axis in axes)
            interpol = jsp.interpolate.RegularGridInterpolator(
                device_axes, field, fill_value=0.0, method="linear"
            )
            inside = level.contains(points)
            values = jnp.where(inside, interpol(points), 0.0)
            samples.append(values.reshape(shape[0], shape[1]))
        return tuple(samples)

    def integrate_component(self, component_field):
        """Integrate one regular field or all missing-scale fields along LOS.

        For nested geometry, every band is integrated with its own path spacing.
        Only the reduced per-scan contributions are moved to one device for the
        final sum; the three-dimensional level fields remain on their devices.

        Parameters
        ----------
        component_field : jax.Array, sequence of jax.Array, or NestedBox
            Field representation accepted by :meth:`scan_component`.

        Returns
        -------
        total : jax.Array, shape (n_scans,)
            Sum of all integrated contributions.
        contributions : tuple of jax.Array
            Per-level integrated contributions. A regular grid returns a
            one-element tuple. Nested arrays have already been reduced over
            path length before being gathered onto the sum device.
        """
        samples = self.scan_component(component_field)
        if not isinstance(self.grid_wsp, NestedGridWorkspace):
            contribution = jnp.trapezoid(
                samples, x=self.los_obj[:, :, 3], axis=1
            )
            return contribution, (contribution,)

        contributions = []
        for values, los in zip(samples, self.los_levels):
            device = next(iter(values.devices()))
            radii = jax.device_put(los[:, :, 3], device)
            contributions.append(jnp.trapezoid(values, x=radii, axis=1))

        sum_device = jax.local_devices()[0]
        gathered = tuple(
            jax.device_put(jax.device_get(contribution), sum_device)
            for contribution in contributions
        )
        return sum(gathered, jnp.zeros_like(gathered[0])), gathered
