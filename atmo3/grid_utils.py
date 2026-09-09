"""Shared real-space and Fourier-space geometry for atmospheric grids."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def _vector3(value, name: str, dtype) -> np.ndarray:
    """Normalize a scalar or length-three input to a NumPy three-vector.

    ``name`` is used in validation errors and ``dtype`` controls the returned
    host-side representation used while constructing static grid shapes.
    """
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.repeat(array, 3)
    elif array.ndim == 1 and array.size == 1:
        array = np.repeat(array, 3)
    elif array.ndim != 1 or array.size != 3:
        raise ValueError(f"{name} must be a scalar or a sequence of three values.")
    return array


class GridWorkspace:
    """Geometry and Fourier-grid utilities for one rectangular 3-D grid.

    Geometry is expressed in one common physical coordinate system. ``origin``
    is the lower box corner, while axes contain sample locations separated by
    ``cell_size``. FFT shapes are stored as Python tuples so JAX can treat them
    as static compilation arguments.
    """

    def __init__(
        self,
        N,
        Lbox=None,
        cell_size=None,
        site_altitude: float = 0.0,
        origin=None,
    ):
        """Construct and validate a regular rectangular grid.

        Parameters
        ----------
        N : int or sequence of three int
            Cell counts. A scalar is broadcast to all dimensions.
        Lbox : float or sequence of three float, optional
            Physical dimensions. Used only if ``cell_size`` is absent.
        cell_size : float or sequence of three float, optional
            Cell dimensions. When both geometry descriptions are supplied,
            this takes precedence and ``Lbox`` is recomputed as ``N*cell_size``.
        site_altitude : float
            Default absolute z coordinate of the lower face.
        origin : sequence of three float, optional
            Absolute lower corner. Defaults to ``(0, 0, site_altitude)``.

        Attributes
        ----------
        dk, d3k : jax.Array
            Per-axis Fourier spacing and Fourier-cell volume.
        nyquist_k : jax.Array, shape (3,)
            Per-axis Nyquist wavenumbers.
        isotropic_kmax : jax.Array
            Radius of the largest fully represented isotropic Fourier sphere.
        rshape, cshape : tuple of int
            Real and half-complex FFT array shapes.
        """
        n_array = _vector3(N, "N", np.int64)
        if np.any(n_array <= 0):
            raise ValueError("Every grid dimension in N must be positive.")

        if cell_size is not None:
            spacing = _vector3(cell_size, "cell_size", np.float64)
            lengths = n_array * spacing
        elif Lbox is not None:
            lengths = _vector3(Lbox, "Lbox", np.float64)
            spacing = lengths / n_array
        else:
            raise ValueError("Either Lbox or cell_size must be provided.")

        if np.any(spacing <= 0.0) or np.any(lengths <= 0.0):
            raise ValueError("Lbox and cell_size values must be positive.")

        if origin is None:
            origin_array = np.array([0.0, 0.0, site_altitude], dtype=np.float64)
        else:
            origin_array = _vector3(origin, "origin", np.float64)

        self.N = jnp.asarray(n_array)
        self.Lbox = jnp.asarray(lengths)
        self.cell_size = jnp.asarray(spacing)
        self.grid_spacing = self.cell_size  # Backwards-compatible name.
        self.site_altitude = float(site_altitude)
        self.origin = jnp.asarray(origin_array)

        self.dk = 2.0 * jnp.pi / self.Lbox
        self.d3k = jnp.prod(self.dk)
        self.nyquist_k = jnp.pi / self.cell_size
        self.isotropic_kmax = jnp.min(self.nyquist_k)

        self.rshape = tuple(int(value) for value in n_array)
        self.cshape = (self.rshape[0], self.rshape[1], self.rshape[2] // 2 + 1)
        self.lower_bounds = self.origin
        self.upper_bounds = self.origin + self.Lbox

    @staticmethod
    @partial(jax.jit, static_argnames=("N_i", "r"))
    def _jit_k_axis(dk_i, N_i, r=False):
        """Build one jitted FFT-frequency axis in angular-wavenumber units."""
        if r:
            return jnp.fft.rfftfreq(N_i) * dk_i * N_i
        return jnp.fft.fftfreq(N_i) * dk_i * N_i

    def k_axis(self, axis=0, r=False, slab_axis=False):
        """Return one Fourier axis in angular-wavenumber units.

        Parameters
        ----------
        axis : int
            Spatial axis index.
        r : bool
            Use the nonnegative rFFT convention, normally for the last axis.
        slab_axis : bool
            Compatibility alias selecting axis 1.
        """
        if slab_axis:
            axis = 1
        return self._jit_k_axis(self.dk[axis], self.rshape[axis], r)

    def k_square(self, kx=None, ky=None, kz=None):
        """Return ``kx**2 + ky**2 + kz**2`` on the half-complex grid."""
        if kx is None:
            kx = self.k_axis(0)
        if ky is None:
            ky = self.k_axis(1)
        if kz is None:
            kz = self.k_axis(2, r=True)
        kxa, kya, kza = jnp.meshgrid(kx, ky, kz, indexing="ij")
        return kxa**2 + kya**2 + kza**2

    def k_magnitude_grid(self):
        """Return radial wavenumber ``|k|`` on the half-complex grid."""
        return jnp.sqrt(self.k_square())

    def band_mask(self, k_low=0.0, k_high=None):
        """Return a radial Fourier mask ``k_low < |k| <= k_high``.

        A nonpositive lower bound includes the DC location. If ``k_high`` is
        omitted, the conservative isotropic Nyquist cutoff is used.
        """
        if k_high is None:
            k_high = self.isotropic_kmax
        magnitude = self.k_magnitude_grid()
        lower = magnitude >= 0.0 if float(k_low) <= 0.0 else magnitude > k_low
        return lower & (magnitude <= k_high)

    @staticmethod
    @partial(jax.jit, static_argnames=("N", "cshape"))
    def _jit_interp2kgrid(dk, k_1d, f_1d, N, cshape):
        """Interpolate a radial one-dimensional function onto an rFFT grid."""
        kx = GridWorkspace._jit_k_axis(dk[0], N[0])
        ky = GridWorkspace._jit_k_axis(dk[1], N[1])
        kz = GridWorkspace._jit_k_axis(dk[2], N[2], r=True)
        kxa, kya, kza = jnp.meshgrid(kx, ky, kz, indexing="ij")
        magnitude = jnp.sqrt(kxa**2 + kya**2 + kza**2).ravel()
        values = jnp.interp(magnitude, k_1d, f_1d, left=0.0, right=0.0)
        return jnp.reshape(values, cshape)

    def interp2kgrid(self, k_1d, f_1d):
        """Interpolate tabulated ``f(k)`` onto this grid's radial modes.

        Modes outside the supplied ``k_1d`` range receive zero.
        """
        return self._jit_interp2kgrid(
            self.dk,
            jnp.asarray(k_1d),
            jnp.asarray(f_1d),
            self.rshape,
            self.cshape,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=("N_i",))
    def _jit_grid_axis(grid_spacing_i, origin_i, N_i):
        """Build one jitted origin-aware real-space coordinate axis."""
        return origin_i + jnp.arange(N_i) * grid_spacing_i

    def grid_axis(self, axis, altitude_axis=False):
        """Return cell coordinates in the common physical coordinate system.

        ``altitude_axis`` remains accepted for compatibility. Coordinates are
        always origin-aware, so it no longer changes the result.
        """
        del altitude_axis
        return self._jit_grid_axis(
            self.cell_size[axis], self.origin[axis], self.rshape[axis]
        )

    def local_grid_axis(self, axis):
        """Return one real-space axis relative to this grid's lower corner."""
        return jnp.arange(self.rshape[axis]) * self.cell_size[axis]

    def axes(self):
        """Return the three global real-space coordinate axes as a tuple."""
        return tuple(self.grid_axis(axis) for axis in range(3))

    @staticmethod
    @partial(jax.jit, static_argnames=("N", "rshape"))
    def _jit_interp2grid(grid_spacing, origin, x_1d, f_1d, N, rshape):
        """Interpolate a vertical profile and broadcast it over x and y."""
        z = GridWorkspace._jit_grid_axis(grid_spacing[2], origin[2], N[2])
        vertical = jnp.interp(z, x_1d, f_1d, left="extrapolate", right="extrapolate")
        return jnp.broadcast_to(vertical.reshape(1, 1, N[2]), rshape)

    def interp2grid(self, x_1d, f_1d):
        """Interpolate an altitude profile onto the full real-space grid."""
        return self._jit_interp2grid(
            self.cell_size,
            self.origin,
            jnp.asarray(x_1d),
            jnp.asarray(f_1d),
            self.rshape,
            self.rshape,
        )

    def contains(self, points):
        """Return whether points lie in this workspace's half-open bounds.

        ``points`` may have any leading shape followed by a final coordinate
        dimension of length three. The result has the corresponding leading
        shape.
        """
        points = jnp.asarray(points)
        return jnp.all(
            (points >= self.lower_bounds) & (points < self.upper_bounds), axis=-1
        )

    def physical_to_index(self, points):
        """Convert global physical coordinates to fractional array indices."""
        return (jnp.asarray(points) - self.origin) / self.cell_size

    def index_to_physical(self, indices):
        """Convert fractional array indices to global physical coordinates."""
        return self.origin + jnp.asarray(indices) * self.cell_size


class NestedGridWorkspace:
    """Shared geometry for a hierarchy of nested rectangular grids.

    Every level has the same array shape but covers a smaller physical volume.
    Children are centered on the telescope in x/y and share the ground-level z
    origin. The class owns geometry only; it stores no field realization.
    """

    def __init__(
        self,
        N,
        nlevels: int | None = None,
        Lbox=None,
        cell_size=None,
        refinement_factor=2,
        telescope_position=None,
        site_altitude: float = 0.0,
    ):
        """Construct a nested grid hierarchy and its disjoint spectral bands.

        Parameters
        ----------
        N : int or sequence of three int
            Cell counts shared by all levels.
        nlevels : int, optional
            Number of boxes including the coarse box.
        Lbox : sequence of three float, optional
            Coarse physical dimensions used with ``refinement_factor``.
        cell_size : sequence, optional
            Explicit per-level spacing. Shape ``(nlevels,)`` is isotropic and
            shape ``(nlevels, 3)`` is anisotropic. It overrides ``Lbox`` and
            ``refinement_factor``.
        refinement_factor : float or sequence of three float
            Box-length reduction from one level to the next.
        telescope_position : sequence of three float, optional
            x/y center of every grid. Its z component is descriptive; lower z
            faces are fixed to ``site_altitude``.
        site_altitude : float
            Absolute lower z boundary of all levels.
        Attributes
        ----------
        levels : tuple of GridWorkspace
            Ordered coarse-to-fine grid geometries.
        band_limits : tuple of tuple
            Effective radial ranges ``(k_low, k_high)``. Positive lower limits
            are open and upper limits are closed.
        """
        if nlevels is None or int(nlevels) <= 0:
            raise ValueError("nlevels must be a positive integer.")

        self.nlevels = int(nlevels)
        n_array = _vector3(N, "N", np.int64)

        if cell_size is not None:
            spacings = self._level_cell_sizes(cell_size, self.nlevels)
            lengths = spacings * n_array[None, :]
        else:
            if Lbox is None:
                raise ValueError("Either Lbox or cell_size must be provided.")
            coarse_lengths = _vector3(Lbox, "Lbox", np.float64)
            factors = _vector3(refinement_factor, "refinement_factor", np.float64)
            if np.any(factors <= 1.0) and self.nlevels > 1:
                raise ValueError("refinement_factor must exceed one on every axis.")
            powers = np.arange(self.nlevels, dtype=np.float64)[:, None]
            lengths = coarse_lengths[None, :] / factors[None, :] ** powers
            spacings = lengths / n_array[None, :]

        if np.any(spacings <= 0.0):
            raise ValueError("All refinement cell sizes must be positive.")

        if telescope_position is None:
            telescope = np.array(
                [lengths[0, 0] / 2.0, lengths[0, 1] / 2.0, site_altitude],
                dtype=np.float64,
            )
        else:
            telescope = _vector3(
                telescope_position, "telescope_position", np.float64
            )

        self.telescope_position = jnp.asarray(telescope)
        self.site_altitude = float(site_altitude)
        levels = []
        for level in range(self.nlevels):
            origin = np.array(
                [
                    telescope[0] - lengths[level, 0] / 2.0,
                    telescope[1] - lengths[level, 1] / 2.0,
                    site_altitude,
                ]
            )
            levels.append(
                GridWorkspace(
                    N=n_array,
                    cell_size=spacings[level],
                    site_altitude=site_altitude,
                    origin=origin,
                )
            )
        self.levels = tuple(levels)

        for parent, child in zip(self.levels[:-1], self.levels[1:]):
            contained = np.all(
                np.asarray(child.lower_bounds) >= np.asarray(parent.lower_bounds)
            ) and np.all(
                np.asarray(child.upper_bounds) <= np.asarray(parent.upper_bounds)
            )
            if not contained:
                raise ValueError("Every refined grid must fit inside its parent.")

        kmax = np.asarray([float(level.isotropic_kmax) for level in self.levels])
        if np.any(np.diff(kmax) <= 0.0):
            raise ValueError(
                "Each refinement must have a larger isotropic Nyquist cutoff."
            )
        self.band_limits = tuple(
            (0.0 if index == 0 else float(kmax[index - 1]), float(kmax[index]))
            for index in range(self.nlevels)
        )

    @staticmethod
    def _level_cell_sizes(cell_size, nlevels: int) -> np.ndarray:
        """Validate and expand explicit per-level cell-size input.

        A one-dimensional input supplies one isotropic value per level; an
        anisotropic input must have the exact shape ``(nlevels, 3)``.
        """
        values = np.asarray(cell_size, dtype=np.float64)
        if values.ndim == 1:
            if values.size != nlevels:
                raise ValueError(
                    "A one-dimensional cell_size must have one value per level."
                )
            values = np.repeat(values[:, None], 3, axis=1)
        elif values.ndim == 2 and values.shape == (nlevels, 3):
            pass
        else:
            raise ValueError(
                "cell_size must have shape (nlevels,) or (nlevels, 3)."
            )
        return values

    @property
    def coarse(self):
        """Return the outermost, lowest-resolution workspace."""
        return self.levels[0]

    @property
    def finest(self):
        """Return the innermost, highest-resolution workspace."""
        return self.levels[-1]

    @property
    def N(self):
        """Return the cell counts shared by all levels."""
        return self.coarse.N

    @property
    def Lbox(self):
        """Return the coarse box dimensions for compatibility."""
        return self.coarse.Lbox

    @property
    def origin(self):
        """Return the coarse box origin for compatibility."""
        return self.coarse.origin

    @property
    def lower_bounds(self):
        """Return the coarse hierarchy's lower physical bounds."""
        return self.coarse.lower_bounds

    @property
    def upper_bounds(self):
        """Return the coarse hierarchy's upper physical bounds."""
        return self.coarse.upper_bounds

    def level(self, index: int) -> GridWorkspace:
        """Return the workspace at a coarse-to-fine level index."""
        return self.levels[index]

    def levels_containing(self, points):
        """Return a final-axis mask indicating which levels contain points."""
        return jnp.stack([level.contains(points) for level in self.levels], axis=-1)

    def finest_level_containing(self, points):
        """Return the finest containing level index, or ``-1`` if outside."""
        contained = self.levels_containing(points)
        indices = jnp.arange(self.nlevels)
        return jnp.max(jnp.where(contained, indices, -1), axis=-1)

    def composite_axis(self, axis: int):
        """Return the sorted union of all sample coordinates along one axis.

        This is a geometry convenience for visualization and diagnostics. LOS
        integration intentionally uses each level's own axis instead.
        """
        values = np.concatenate(
            [np.asarray(level.grid_axis(axis)) for level in self.levels]
        )
        return jnp.asarray(np.unique(values))

    def band_mask(self, level: int):
        """Return the disjoint radial Fourier mask assigned to one level."""
        low, high = self.band_limits[level]
        return self.levels[level].band_mask(low, high)
