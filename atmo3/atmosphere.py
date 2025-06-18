from . import cube
from . import grid_utils as gutl

class Atmosphere:
    
    def __init__(self, 
                 nside_grid: int = 128, 
                 box_length_in_m: float = 10000.0
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
        self.grid_wsp        = gutl.GridWorkspace(N=self.N, Lbox=self.Lbox)
        self.component_names = []
        self.components      = {}
        
    
    def add_component(
        self,
        field_name: str,
        field_unit: str,
        pspec: dict,
        rescale: dict,
        seed: int,
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
            seed=seed,
            nsub=nsub
        )
        
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
                
    def compute_emission(self):
        pass