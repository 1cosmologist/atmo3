import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.spatial.transform import Rotation
import astropy.units as u

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

def polygonal_vertices(polygon_order, fwhm: u.Quantity):
    """
    Generate the vertices of a regular polygon on the unit sphere corresponding to the given FWHM.
    Parameters:
    polygon_order : int
        Number of vertices of the polygon.
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the polygon vertices on the unit sphere.
    """
    angles = jnp.linspace(0, 2 * jnp.pi, polygon_order, endpoint=False)
    if fwhm.unit.is_equivalent(u.degree):
        fwhm = fwhm.to(u.radian)
    cos = jnp.cos(fwhm.to(u.radian).value)
    sin = jnp.sin(fwhm.to(u.radian).value)
    return jnp.array(
        [cos * jnp.ones_like(angles), sin * jnp.cos(angles), sin * jnp.sin(angles)]
    ).T

def hexagon_first_rim(fwhm: u.Quantity):
    """
    Generate the vertices of the first rim of a hexagon on the unit sphere corresponding to the given FWHM.
    Parameters:
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    """
    return polygonal_vertices(6, fwhm)

def hexagon_second_rim(fwhm: u.Quantity):
    """
    Generate the vertices of the second rim of a hexagon on the unit sphere corresponding to twice the given FWHM.
    Parameters:
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    """
    vertices = polygonal_vertices(6, 2 * fwhm)
    ## add midpoints between two consecutive vertices
    midpoints = (vertices + jnp.roll(vertices, -1, axis=0)) / 2
    return jnp.vstack([vertices, midpoints])

def hexagon_center_and_first_rim(fwhm: u.Quantity):
    """
    Generate the vertices of the center and first rim of a hexagon on the unit sphere corresponding to the given FWHM.
    Parameters:
    fwhm : u.Quantity
        Full width at half maximum angle, specifying the radius of the hexagon vertices on the unit sphere.
    """
    center = jnp.array([[1.0, 0.0, 0.0]])
    first_rim = hexagon_first_rim(fwhm)
    return jnp.vstack([center, first_rim])

def unit_vectors_center_and_first_rim(
    fwhm: u.Quantity, elevation_in_deg, azimuth_in_deg
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
    center_and_first_rim = hexagon_center_and_first_rim(fwhm)
    rotated_fp_center_and_first_rim = rot_matrix.apply(center_and_first_rim)
    return rotated_fp_center_and_first_rim

def los_points_coords_radius(
    site_altitude,
    Lbox,
    altitude_slice, 
    unit_vector, 
    det_pos,
    west_wind: float = None,
    south_wind: float = None,
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
        mask_r = r < Lbox
    else:
        mask_r = jnp.ones_like(r, dtype=bool)
    if west_wind is not None and south_wind is not None:
        # Apply wind correction if west and south winds are provided
        x_los += west_wind * delta_t
        y_los += south_wind * delta_t
    return jnp.array([x_los, y_los, altitude_slice, r, mask_r]).T

def los_points_center_and_first_rim(
    altitude_slice,
    fwhm: u.Quantity,
    elevation_in_deg,
    azimuth_in_deg,
    detector_position: jnp.ndarray,
    west_wind: float = None,
    south_wind: float = None,
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
        fwhm, elevation_in_deg, azimuth_in_deg
    )
    los_center_and_first_rim = jax.vmap(
        lambda uv: los_points_coords_radius(
            altitude_slice, uv, detector_position, west_wind=west_wind, south_wind=south_wind, delta_t=delta_t, max_radius=max_radius
        )
    )(rotated_fp_center_and_first_rim)
    return los_center_and_first_rim