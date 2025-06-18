import cube
import atmo3.grid_utils as ffu

class Atmosphere:
    
    def __init__(self, 
                 nside_grid: int = 128, 
                 box_length_in_m: float = 10000.0
    ) -> None:
        
        self.N               = nside_grid
        self.Lbox            = box_length_in_m
        self.fft_wsp         = ffu.FFT_spec(N=self.N, Lbox=self.Lbox)
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
        
        self.component_names.append(field_name)
        self.components[field_name] = cube.Cube(
            N=self.N,
            Lbox=self.Lbox,
            partype='jaxshard',
            fft_wsp=self.fft_wsp,
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
        
        if component_name in self.component_names:
            self.components[component_name].generate_field_realization(time_step=time_step)
        else:
            for component in self.components.values():
                component.generate_field_realization(time_step=time_step)