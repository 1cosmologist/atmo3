from . import cube
from . import grid_utils as gutl
from . import constants  as const
import jax
import jax.numpy as jnp
import numpy as np
import gc

class Atmosphere:
    
    def __init__(self, 
                 nside_grid: int = 128, 
                 box_length_in_m: float = 10000.0,
                 site_altitude: float = 0.0
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
        self.N               = nside_grid
        self.Lbox            = box_length_in_m
        self.site_altitude   = site_altitude
        self.grid_wsp        = gutl.GridWorkspace(
                                                N=self.N, 
                                                Lbox=self.Lbox, 
                                                site_altitude=self.site_altitude
                                                )
        self.component_names  = []
        self.components       = {}
        self.properties_names = []
        self.properties       = {}

    def add_component(
        self,
        field_name: str,
        field_unit: str,
        pspec: dict,
        rescale: dict,
        seed: int,
        mean: dict = None,
        nsub: int = 1024**3
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
            nsub=nsub
        )

    def add_property(
        self,
        property_name: str,
        property_unit: str,
        property_value: dict
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
            "value": property_value
        }

    def generate_realization(
        self,
        time_step: int = 0,
        component_name: str = None
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
            self.components[component_name].generate_field_realization(time_step=time_step)
        else:
            for component in self.components.values():
                component.generate_field_realization(time_step=time_step)
                
    def compute_virtual_temperature(self):
        if not (('specific humidity' in self.component_names) and ('temperature' in self.properties_names)):
            raise ValueError("Both 'specific humidity' and 'temperature' components must be present to compute virtual temperature.")
        
        self.component_names.append('virtual temperature')
        self.components['virtual temperature'] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            grid_wsp=self.grid_wsp,
            field_name='virtual temperature',
            field_unit='K',
            pspec=None,
            rescale=None,
            seed=None,
            nsub=None
        )
        self.components['virtual temperature'].field = self.grid_wsp.interp2grid(self.properties['temperature']['value']['h'], self.properties['temperature']['value']['f']) * (1.0 + 0.61 * self.components['specific humidity'].field)

    def compute_pressure(self, P_surface: float = 55500.):
        if 'virtual temperature' not in self.component_names:
            raise ValueError("'virtual temperature' component must be present to compute pressure.")
        
        self.component_names.append('pressure')
        self.components['pressure'] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            grid_wsp=self.grid_wsp,
            field_name='pressure',
            field_unit='Pa',
            pspec=None,
            rescale=None,
            seed=None,
            nsub=None
        )

        # integrand = const.g / const.R_dry_air / self.components['virtual temperature'].field
        # z_axis = self.grid_wsp.grid_axis(altitude_axis=True)
        ## FIX-ME: Replace with jax native intergral by avoiding slicing
        # integral = np.zeros(self.grid_wsp.rshape_local)
        
        # for i in range(self.N):
        #     integral[:,:,i] = np.asarray(jnp.trapezoid(integrand[:,:,0:i+1], x=z_axis[0:i+1], axis=2))
            
        # integral = jnp.asarray(integral)
        ###
        
        # del integrand, z_axis ; gc.collect()
        self.components['pressure'].field = P_surface * jnp.exp(-(const.g / const.R_dry_air) * jnp.cumsum(self.grid_wsp.grid_spacing / self.components['virtual temperature'].field, axis=2))
        # del integrand; gc.collect()

    def compute_water_vapor_density(self):
        if not (('specific humidity' in self.component_names) and ('pressure' in self.component_names) and ('virtual temperature' in self.component_names)):
            raise ValueError("Components 'specific humidity', 'pressure', and 'virtual temperature' must be present to compute water vapor density.")
        
        self.component_names.append('water vapor density')
        self.components['water vapor density'] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            grid_wsp=self.grid_wsp,
            field_name='water vapor density',
            field_unit='kg m-3',
            pspec=None,
            rescale=None,
            seed=None,
            nsub=None
        )
        self.components['water vapor density'].field = self.components['specific humidity'].field * self.components['pressure'].field / const.R_dry_air / self.components['virtual temperature'].field
    
    def compute_pwv(self):
        z_axis = self.grid_wsp.grid_axis(altitude_axis=True)
        
        self.add_property(
            property_name='precipitable water vapor',
            property_unit='mm',
            property_value={
                'h': self.site_altitude,
                'f': jnp.trapezoid(self.components['water vapor density'].field, x=z_axis, axis=2)  # Assuming 1 kg m-2 = 1 mm of PWV
            }
        )
        
    def compute_emission(self):
        pass