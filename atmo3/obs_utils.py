import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.spatial.transform import Rotation
# import astropy.units as u

### Analytical line-of-sight
### These functions are used to interpolate values at the exact coordinates given an atmo3 realization.

def rotation_matrix(elevation_in_deg, azimuth_in_deg):
    """
    Create a rotation matrix from elevation and azimuth angles in degrees.
    This transforms coordinates in the focalplane frame to the global frame.
    Parameters:
    elevation_in_deg : float
        Elevation angle in degrees.
    azimuth_in_deg : float
        Azimuth angle in degrees.
    """
    rot_y = Rotation.from_euler("y", -elevation_in_deg, degrees=True)
    rot_z = Rotation.from_euler("z", azimuth_in_deg, degrees=True)
    total_rotation = rot_z * rot_y
    return total_rotation


def polygonal_vertices(polygon_order, fwhm_in_arcmin):
    """
    Generate the vertices of a regular polygon on the unit sphere corresponding to the given FWHM.
    Parameters:
    polygon_order : int
        Number of vertices of the polygon.
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the polygon vertices on the unit sphere.
    """
    angles = jnp.linspace(0, 2 * jnp.pi, polygon_order, endpoint=False)
    
    fwhm = jnp.deg2rad(fwhm_in_arcmin / 60.)
    cos = jnp.cos(fwhm)
    sin = jnp.sin(fwhm)
    return jnp.array(
        [cos * jnp.ones_like(angles), sin * jnp.cos(angles), sin * jnp.sin(angles)]
    ).T


def hexagon_first_rim(fwhm_in_arcmin):
    """
    Generate the vertices of the first rim of a hexagon on the unit sphere corresponding to the given FWHM.
    Parameters:
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    """
    return polygonal_vertices(6, fwhm_in_arcmin)


def hexagon_second_rim(fwhm_in_arcmin):
    """
    Generate the vertices of the second rim of a hexagon on the unit sphere corresponding to twice the given FWHM.
    Parameters:
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    """
    vertices = polygonal_vertices(6, 2 * fwhm_in_arcmin)
    ## add midpoints between two consecutive vertices
    midpoints = (vertices + jnp.roll(vertices, -1, axis=0)) / 2
    return jnp.vstack([vertices, midpoints])


def hexagon_center_and_first_rim(fwhm_in_arcmin):
    """
    Generate the vertices of the center and first rim of a hexagon on the unit sphere corresponding to the given FWHM.
    Parameters:
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    """
    center = jnp.array([[1.0, 0.0, 0.0]])   ## ??
    
    first_rim = hexagon_first_rim(fwhm_in_arcmin)
    return jnp.vstack([center, first_rim])


def unit_vectors_center_and_first_rim(
    fwhm_in_arcmin, elevation_in_deg, azimuth_in_deg
):
    """
    Generate the pointing vectors for the center and first rim of a hexagon pointing in a given direction.
    Parameters:
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    elevation_in_deg : float
        Elevation angle in degrees.
    azimuth_in_deg : float
        Azimuth angle in degrees.
    """
    rot_matrix = rotation_matrix(elevation_in_deg, azimuth_in_deg)
    center_and_first_rim = hexagon_center_and_first_rim(fwhm_in_arcmin)
    rotated_fp_center_and_first_rim = rot_matrix.apply(center_and_first_rim)
    return rotated_fp_center_and_first_rim


def los_points_coords_radius(
    site_altitude,
    Lbox,
    altitude_slice, 
    unit_vector, 
    det_pos,
    north_wind: float = None,
    east_wind: float = None,
    delta_t: float = 0.0, 
    max_radius: bool = False
):
    """
    Calculate the line-of-sight points coordinates and radius for a given altitude slice, pointing vector, and detector position.
    Parameters:
    altitude_slice : float
        Altitude slice value.
    unit_vector : jnp.ndarray
        Unit vector representing the direction.
    det_pos : jnp.ndarray
        Detector position coordinates.
    west_wind : float, optional
        West wind speed in m/s (default is None).
    south_wind : float, optional
        South wind speed in m/s (default is None).
    delta_t : float, optional
        Time step in seconds after generation of the cube (default is 0.0).
    max_radius : bool, optional
        Whether to apply a maximum radius mask (default is False).
    """
    r = (altitude_slice - site_altitude) / unit_vector[2]
    x_los = det_pos[0] + r * unit_vector[0]
    y_los = det_pos[1] + r * unit_vector[1]
    if max_radius:
        mask_r = r < Lbox[2]
    else:
        mask_r = jnp.ones_like(r, dtype=bool)
    if north_wind is not None and east_wind is not None:
        # Apply wind correction if west and south winds are provided

        x_los += east_wind  * delta_t
        y_los += north_wind * delta_t
    return jnp.array([x_los, y_los, altitude_slice, r, mask_r]).T


def los_points_center_and_first_rim(
    altitude_slice,
    fwhm_in_arcmin,
    elevation_in_deg,
    azimuth_in_deg,
    detector_position: jnp.ndarray,
    north_wind: float = None,
    east_wind: float = None,
    delta_t: float = 0.0,
    max_radius: bool = False,
):
    """
    Calculate the line-of-sight points coordinates and radius for the center and first rim of a hexagon at a given altitude slice.
    Parameters:
    altitude_slice : float
        Altitude slice value.
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    elevation_in_deg : float
        Elevation angle in degrees.
    azimuth_in_deg : float
        Azimuth angle in degrees.
    detector_position : jnp.ndarray
        Detector position coordinates.
    west_wind : float, optional
        West wind speed in m/s (default is None).
    south_wind : float, optional
        South wind speed in m/s (default is None).
    delta_t : float, optional
        Time step in seconds after generation of the cube (default is 0.0).
    max_radius : bool, optional
        Whether to apply a maximum radius mask (default is False).
    """
    rotated_fp_center_and_first_rim = unit_vectors_center_and_first_rim(
        fwhm_in_arcmin, elevation_in_deg, azimuth_in_deg
    )
    los_center_and_first_rim = jax.vmap(
        lambda uv: los_points_coords_radius(
            altitude_slice, uv, detector_position, north_wind=north_wind, east_wind=east_wind, delta_t=delta_t, max_radius=max_radius
        )
    )(rotated_fp_center_and_first_rim)
    return los_center_and_first_rim


@jax.jit
def _wind_evolved_layer(field_sheet, north_wind_slice, east_wind_slice, delta_t_sec):
    shift_x = east_wind_slice * delta_t_sec
    shift_y = north_wind_slice * delta_t_sec
    
    return jnp.roll(field_sheet, [shift_x, shift_y], axis=[0,1]) 

@jax.jit
def wind_evolved_field(field, north_wind, east_wind, delta_t_sec):
    return jax.vmap(
        lambda sheet, nw, ew: _wind_evolved_layer(sheet, nw, ew, delta_t_sec),
        in_axes=(2, 0, 0),
        out_axes=2,
    )(field, north_wind, east_wind)