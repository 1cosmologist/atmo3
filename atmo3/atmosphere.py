from . import cube
from . import grid_utils as gutl
from . import constants as const
import jax
import jax.numpy as jnp
import numpy as np
import gc


class Atmosphere:
    def __init__(
        self,
        nside_grid: int = 128,
        box_length_in_m: float = 10000.0,
        site_altitude: float = 0.0,
    ) -> None:
        """
        Initialize the atmosphere object.

        Parameters
        ----------
        nside_grid : int, optional
            Number of cells per side of the grid. Defaults to 128.
        box_length_in_m : float, optional
            Box length in meters. Defaults to 10000.0.

        Attributes
        ----------
        N : int
            Number of cells per side of the grid.
        Lbox : float
            Box length in meters.
        grid_wsp : gutl.GridWorkspace
            Grid workspace object.
        component_names : list
            List of component names.
        components : dict
            Dictionary of component objects.
        """
        self.N = nside_grid
        self.Lbox = box_length_in_m
        self.site_altitude = site_altitude
        self.grid_wsp = gutl.GridWorkspace(
            N=self.N, Lbox=self.Lbox, site_altitude=self.site_altitude
        )
        self.component_names = []
        self.components = {}
        self.properties_names = []
        self.properties = {}

    def add_component(
        self,
        field_name: str,
        field_unit: str,
        pspec: dict,
        rescale: dict,
        seed: int,
        mean: dict = None,
        nsub: int = 1024**3,
    ) -> None:
        """
        Add a component to the atmosphere.

        Parameters
        ----------
        field_name : str
            Name of the component.
        field_unit : str
            Unit of the component.
        pspec : dict
            Power spectrum of the component.
        rescale : dict
            Rescaling factors as a function of height.
        seed : int
            Random seed.
        nsub : int, optional
            Number of subsamples for random number generation. Defaults to 1024**3.
        """

        self.component_names.append(field_name)
        self.components[field_name] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            grid_wsp=self.grid_wsp,
            field_name=field_name,
            field_unit=field_unit,
            pspec=pspec,
            rescale=rescale,
            mean=mean,
            seed=seed,
            nsub=nsub,
        )

    def add_property(
        self, property_name: str, property_unit: str, property_value: dict
    ) -> None:
        """
        Add a property to the atmosphere.

        Parameters
        ----------
        property_name : str
            Name of the property.
        property_unit : str
            Unit of the property.
        property_value : dict
            Property value as a function of height.
        """

        self.properties_names.append(property_name)
        self.properties[property_name] = {
            "unit": property_unit,
            "value": property_value,
        }

    def generate_realization(
        self, time_step: int = 0, component_name: str = None
    ) -> None:
        """
        Generate a realization of the atmospheric component(s).

        Parameters
        ----------
        time_step : int, optional
            Time step of the realization. Defaults to 0.
        component_name : str, optional
            Name of the component to generate. If not given, all components are generated.
        """
        if component_name in self.component_names:
            self.components[component_name].generate_field_realization(
                time_step=time_step
            )
        else:
            for component in self.components.values():
                component.generate_field_realization(time_step=time_step)

    def compute_virtual_temperature(self) -> None:
        """
        Compute virtual temperature from specific humidity and temperature components.

        The virtual temperature is computed as:

        T_v = T * (1.0 + 0.61 * q)

        where T is the temperature, q is the specific humidity, and T_v is the virtual temperature.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If 'specific humidity' and 'temperature' components are not present.
        """
        if not (
            ("specific humidity" in self.component_names)
            and ("temperature" in self.properties_names)
        ):
            raise ValueError(
                "Both 'specific humidity' and 'temperature' components must be present to compute virtual temperature."
            )

        self.component_names.append("virtual temperature")
        self.components["virtual temperature"] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            grid_wsp=self.grid_wsp,
            field_name="virtual temperature",
            field_unit="K",
            pspec=None,
            rescale=None,
            seed=None,
            nsub=None,
        )
        self.components["virtual temperature"].field = self.grid_wsp.interp2grid(
            self.properties["temperature"]["value"]["h"],
            self.properties["temperature"]["value"]["f"],
        ) * (1.0 + 0.61 * self.components["specific humidity"].field)

    def compute_pressure(self, P_surface: float = 55500.0) -> None:
        """
        Compute pressure from virtual temperature component.

        The pressure is computed by integrating the hydrostatic equation from the top of the atmosphere to the bottom, using the virtual temperature as the temperature profile.

        Parameters
        ----------
        P_surface : float, optional
            Surface pressure in Pascal. Defaults to 55500.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If 'virtual temperature' component is not present.
        """
        if "virtual temperature" not in self.component_names:
            raise ValueError(
                "'virtual temperature' component must be present to compute pressure."
            )

        self.component_names.append("pressure")
        self.components["pressure"] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            grid_wsp=self.grid_wsp,
            field_name="pressure",
            field_unit="Pa",
            pspec=None,
            rescale=None,
            seed=None,
            nsub=None,
        )

        self.components["pressure"].field = P_surface * jnp.exp(
            -(const.g / const.R_dry_air)
            * jnp.cumsum(
                self.grid_wsp.grid_spacing
                / self.components["virtual temperature"].field,
                axis=2,
            )
        )

    def compute_water_vapor_density(self) -> None:
        """
        Compute water vapor density from specific humidity, pressure, and virtual temperature components.

        The water vapor density is computed by multiplying the specific humidity, pressure, and virtual temperature components, and then dividing by the dry air gas constant and the virtual temperature.

        Raises
        ------
        ValueError
            If 'specific humidity', 'pressure', and 'virtual temperature' components are not present.
        """
        if not (
            ("specific humidity" in self.component_names)
            and ("pressure" in self.component_names)
            and ("virtual temperature" in self.component_names)
        ):
            raise ValueError(
                "Components 'specific humidity', 'pressure', and 'virtual temperature' must be present to compute water vapor density."
            )

        self.component_names.append("water vapor density")
        self.components["water vapor density"] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            grid_wsp=self.grid_wsp,
            field_name="water vapor density",
            field_unit="kg m-3",
            pspec=None,
            rescale=None,
            seed=None,
            nsub=None,
        )
        self.components["water vapor density"].field = (
            self.components["specific humidity"].field
            * self.components["pressure"].field
            / const.R_dry_air
            / self.components["virtual temperature"].field
        )

    def compute_pwv_along_z(self) -> None:
        """
        Compute the precipitable water vapor (PWV) from the water vapor density component.

        The PWV is computed by integrating the water vapor density field along the altitude axis.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If 'water vapor density' component is not present.

        Notes
        -----
        The PWV is computed by integrating the water vapor density field along the altitude axis, and then converting the result from kg m-2 to mm by assuming 1 kg m-2 = 1 mm of PWV.
        """
        z_axis = self.grid_wsp.grid_axis(altitude_axis=True)

        self.add_property(
            property_name="precipitable water vapor",
            property_unit="mm",
            property_value={
                "h": self.site_altitude,
                "f": jnp.trapezoid(
                    self.components["water vapor density"].field, x=z_axis, axis=2
                ),  # Assuming 1 kg m-2 = 1 mm of PWV
            },
        )

    def compute_pwv_along_los(
        self,
        detector_position: jnp.ndarray,
        pointing: jnp.ndarray,
        kmin: int = None,
        kmax: int = None,
        max_radius: float = None,
    ) -> None:
        """
        Compute the precipitable water vapor (PWV) along line of sight.

        Parameters
        ----------
        detector_position : jnp.ndarray
            Position of the detector in meters.
        pointing : jnp.ndarray
            Pointing direction of the detector in degrees.
        max_radius : float
            Maximum radius from the detector to consider in meters.

        Returns
        -------
        None
        """

        theta, phi, psi = pointing
        unit_pointing_vec = self.grid_wsp.lonlat_to_unitvec(theta, phi, lonlat=True)

        selected_voxels = self.grid_wsp.zlice_to_selected_voxels_along_los(
            unit_pointing_vec_reference=unit_pointing_vec,
            kmin=kmin,
            kmax=kmax,
            detector_position=detector_position,
            max_radius=max_radius,
        )

        # Compute radial distances and extract water vapor density values
        radius = jnp.linalg.norm(
            selected_voxels * self.grid_wsp.grid_spacing
            - detector_position[:, jnp.newaxis],
            axis=0,
        )

        wv_cube = self.components["water vapor density"].field
        f = wv_cube[
            (selected_voxels[0]).astype(int),
            (selected_voxels[1]).astype(int),
            (selected_voxels[2]).astype(int),
        ]

        self.add_property(
            property_name="precipitable water vapor",
            property_unit="mm",
            property_value={
                "h": self.site_altitude,
                "Direction": pointing,
                "f": jnp.trapezoid(f, radius),  # Assuming 1 kg m-2 = 1 mm of PWV
            },
        )

    def compute_emission(self):
        pass
