import jax
import jax.numpy as jnp
import gc

gamma_tol_default = 10.0  # degrees
default_tol = jnp.cos(jnp.deg2rad(gamma_tol_default))


class GridWorkspace:
    def __init__(self, N, Lbox, site_altitude=0.0, partition="jaxshard"):
        """
        Initialize the grid workspace object.

        Parameters
        ----------
        N : int
            Number of cells per side of the grid.
        Lbox : float
            Box length in meters.
        partition : str, optional
            Type of partitioning. Defaults to 'jaxshard'.

        Attributes
        ----------
        N : int
            Number of cells per side of the grid.
        Lbox : float
            Box length in meters.
        dk : float
            Grid spacing in k-space.
        d3k : float
            Grid spacing in k-space cubed.
        rshape : tuple
            Shape of a real-space array.
        cshape : tuple
            Shape of a complex-space array.
        rshape_local : tuple
            Shape of a local real-space array.
        cshape_local : tuple
            Shape of a local complex-space array.
        start : int
            Starting index of the local array.
        end : int
            Ending index of the local array.
        grid_spacing : float
            Grid spacing in real-space.
        field : None
            Placeholder for the field array.
        ngpus : int
            Number of GPUs.
        host_id : int
            Host ID.
        parttype : str
            Type of partitioning.
        partaxis : int
            Axis to partition the array.
        """
        self.N = N
        self.Lbox = Lbox
        self.site_altitude = (
            site_altitude  # Default altitude, can be set later if needed
        )

        self.dk = 2 * jnp.pi / self.Lbox
        self.d3k = self.dk * self.dk * self.dk

        self.rshape = (self.N, self.N, self.N)
        self.cshape = (self.N, self.N, self.N // 2 + 1)
        self.rshape_local = (self.N, self.N, self.N)
        self.cshape_local = (self.N, self.N, self.N // 2 + 1)

        self.start = 0
        self.end = self.N

        self.grid_spacing = self.Lbox / self.N

        self.field = None

        # needed for running on CPU with a single process
        self.ngpus = 1
        self.host_id = 0

        self.parttype = partition

        if self.parttype == "jaxshard":
            self.ngpus = jax.device_count()
            self.host_id = jax.process_index()
            self.start = self.host_id * self.N // self.ngpus
            self.end = (self.host_id + 1) * self.N // self.ngpus

            self.partaxis = 1
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)

    def k_axis(self, r=False, slab_axis=False):
        if r:
            k_i = jnp.fft.rfftfreq(self.N) * self.dk * self.N  # .astype(jnp.float32)
        else:
            k_i = jnp.fft.fftfreq(self.N) * self.dk * self.N  # .astype(jnp.float32)
        if slab_axis:
            return k_i[self.start : self.end]  # .astype(jnp.float32)
        return k_i

    def k_square(self, kx, ky, kz):
        kxa, kya, kza = jnp.meshgrid(kx, ky, kz, indexing="ij")
        del kx, ky, kz
        gc.collect()

        k2 = kxa**2 + kya**2 + kza**2  # .astype(jnp.float32)
        del kxa, kya, kza
        gc.collect()

        return k2

    def interp2kgrid(self, k_1d, f_1d):
        kx = self.k_axis()
        ky = self.k_axis(slab_axis=True)
        kz = self.k_axis(r=True)

        interp_fcn = jnp.sqrt(self.k_square(kx, ky, kz)).ravel()
        del kx, ky, kz
        gc.collect()

        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left=0.0, right="extrapolate")
        return jnp.reshape(interp_fcn, self.cshape_local)  # .astype(jnp.float32)

    def grid_axis(self, slab_axis=False, altitude_axis=False):
        x_i = jnp.arange(self.N) * self.grid_spacing  # .astype(jnp.float32)

        if altitude_axis:
            x_i = x_i + self.site_altitude

        if slab_axis:
            return x_i[self.start : self.end]
        return x_i

    def interp2grid(self, x_1d, f_1d):
        x = self.grid_axis()
        y = self.grid_axis(slab_axis=True)
        z = self.grid_axis(altitude_axis=True)

        xx, yy, zz = jnp.meshgrid(x, y, z, indexing="ij")

        del x, y, z, xx, yy
        gc.collect()

        zz = zz.ravel()
        interp_fcn = jnp.interp(zz, x_1d, f_1d, left=0.0, right="extrapolate")
        return jnp.reshape(interp_fcn, self.rshape_local)  # .astype(jnp.float32)

    def pointing_vec_center_voxel(
        self,
        indices_grid: jnp.ndarray,
        detector_position: jnp.ndarray = None,
    ):
        """
        Given the ijk coordinates of a list of voxels, compute the unit pointing vector from a given detector position to the center of each voxel.
        Parameters
        ----------
        indices: jnp.ndarray
            An array of shape (..., 3) with integer voxel indices (i,j,k).
        detector_position : jnp.ndarray, optional
            The position of the detector in meters. Default is [0.0, 0.0, 0.0].
        Returns
        -------
        normalized_coords : jnp.ndarray
            The unit pointing vector from the detector position to the center of the voxel.
        """
        if detector_position is None:
            detector_position = jnp.zeros(3, dtype=jnp.float32)
        # --- Step 1: in-bounds mask (shape (...,1)) ---
        in_bounds = jnp.all(
            (indices_grid >= 0) & (indices_grid < self.N), axis=-1, keepdims=True
        )

        voxel_centers = (indices_grid.astype(jnp.float32) + 0.5) * self.grid_spacing

        pointing_vecs = voxel_centers - detector_position
        norms = jnp.linalg.norm(pointing_vecs, axis=-1, keepdims=True)

        unit_pointing_vecs = pointing_vecs / norms

        # mask invalid outputs with NaN
        return jnp.where(in_bounds & (norms > 0), unit_pointing_vecs, jnp.nan)

    def lonlat_to_unitvec(self, theta: float, phi: float, lonlat: bool = True):
        """
        Convert longitude and latitude to a unit vector.
        Parameters
        ----------
        theta, phi : float
            Angular coordinates of a point on the sphere

        lonlatbool : bool
            If True, input angles are assumed to be longitude and latitude in degree, otherwise, they are co-latitude and longitude in radians.

        Returns
        -------
        jnp.ndarray
            A unit vector corresponding to the given longitude and latitude.
        """
        if lonlat:
            theta = jnp.deg2rad(theta)
            phi = jnp.deg2rad(phi)
            theta = jnp.pi / 2 - theta

        # Compute the unit vector components
        x = jnp.sin(theta) * jnp.cos(phi)
        y = jnp.sin(theta) * jnp.sin(phi)
        z = jnp.cos(theta)

        return jnp.array([x, y, z])

    def separation_distance_pointing_voxel_to_pointing_detector(
        self,
        unit_pointing_vecs: jnp.ndarray,
        unit_pointing_vec_reference: jnp.ndarray,
        tolerance: float = default_tol,
    ) -> jnp.ndarray:
        """
        Given unit pointing vectors from detector to voxels, compute the angular separation between these vectors and a given pointing direction and mask voxels outside the cone.
        Parameters
        ----------
        unit_pointing_vecs : jnp.ndarray
            An array of shape (..., 3) with unit pointing vectors from the detector to the voxels.
        unit_pointing_vec_reference : jnp.ndarray
            A unit vector of shape (3,) representing the reference pointing direction.
        tolerance : float, optional
            Cosine of the angular tolerance in degrees. Default is cos(10.0°)~0.985.
        Returns
        -------
        mask : jnp.ndarray
            A boolean array of the same shape as unit_pointing_vecs[..., 0] indicating which voxels are within the angular tolerance.
        """

        # dot product along last axis (elementwise)
        dot_products = jnp.sum(
            unit_pointing_vecs * unit_pointing_vec_reference, axis=-1
        )

        # mask voxels within angular tolerance
        voxels_within_angle = dot_products >= tolerance

        return voxels_within_angle

    def zlayer_to_voxels(
        self,
        k_layer_center: int,
        unit_pointing_vec_reference: jnp.ndarray,
        detector_position: jnp.ndarray = None,
        tolerance: float = default_tol,
    ):
        """
        For a given altitude layer, return the coordinates of the voxels within a cone defined by a pointing direction and angular tolerance.
        Parameters
        ----------
        k_layer_center : int
            The k index of the layer in altitude corresponding to k_layer_center.
        unit_pointing_vec_reference : jnp.ndarray
            A unit vector of shape (3,) representing the reference pointing direction.
        tolerance : float, optional
            Cosine of the angular tolerance in degrees. Default is cos(10.0°)~0.985.
        detector_position : jnp.ndarray, optional
            The position of the detector in meters relative to the site altitude. Default is [0.0, 0.0, 0.0].
        Returns
        -------
        mask: jnp.ndarray
            A boolean array indicating which voxels are within the angular tolerance.
        """

        i, j = jnp.meshgrid(
            jnp.arange(self.N), jnp.arange(self.N), indexing="ij"
        )  # shape (N,N)
        k = jnp.full_like(i, k_layer_center)  # shape (N,N)

        indices_2d = jnp.stack([i, j, k], axis=-1)  # shape (N,N,3)

        unit_pointing_vecs = self.pointing_vec_center_voxel(
            indices_grid=indices_2d, detector_position=detector_position
        )  # shape (N,N,3)

        voxels_within_angle = (
            self.separation_distance_pointing_voxel_to_pointing_detector(
                unit_pointing_vecs, unit_pointing_vec_reference, tolerance
            )
        )  # shape (N,N)

        masked_indices = jnp.where(voxels_within_angle[..., None], indices_2d, 0)

        return masked_indices

    def zslice_to_voxels(
        self,
        unit_pointing_vec_reference: jnp.ndarray,
        kmin: int = None,
        kmax: int = None,
        detector_position: jnp.ndarray = None,
        tolerance: float = default_tol,
    ):
        """
        For a given altitude slice, return the coordinates of the voxels within a cone defined by a pointing direction and angular tolerance.
        Parameters
        ----------
        kmin, kmax : int
            The k indices defining the slice in altitude. Nslice = kmax - kmin.
        unit_pointing_vec_reference : jnp.ndarray
            A unit vector of shape (3,) representing the reference pointing direction.
        tolerance : float, optional
            Cosine of the angular tolerance in degrees. Default is cos(10.0°)~0.985.
        detector_position : jnp.ndarray, optional
            The position of the detector in meters. Default is [0.0, 0.0, 0.0].
        Returns
        -------
        mask: jnp.ndarray
            A boolean array indicating which voxels are within the angular tolerance.
        """
        if kmin is None:
            kmin = self.start
        if kmax is None:
            kmax = self.end

        k_slice = jnp.arange(kmin, kmax)

        def map_func(k):
            return self.zlayer_to_voxels(
                k_layer_center=k,
                unit_pointing_vec_reference=unit_pointing_vec_reference,
                detector_position=detector_position,
                tolerance=tolerance,
            )

        voxels_indices_slices = jax.vmap(map_func)(k_slice)

        voxels_indices_slices = jnp.swapaxes(voxels_indices_slices, axis1=0, axis2=-1)

        return voxels_indices_slices  # shape (3,N,N,Nslice)

    def intercepted_voxel_with_pointing_detector(
        self, unit_pointing_vecs: jnp.ndarray, unit_pointing_vec_reference: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Given unit pointing vectors from detector to voxels, find the voxel closest to the pointing direction.
        Parameters
        ----------
        unit_pointing_vecs : jnp.ndarray
            An array of shape (..., 3) with unit pointing vectors from the detector to the voxels.
        unit_pointing_vec_reference : jnp.ndarray
            A unit vector of shape (3,) representing the reference pointing direction.
        Returns
        -------
        intercepted_voxel : jnp.ndarray
            The index of the voxel closest to the pointing direction.
        """

        # dot product along last axis (elementwise)
        dot_products = jnp.sum(
            unit_pointing_vecs * unit_pointing_vec_reference, axis=-1
        )

        i, j = jnp.unravel_index(jnp.argmax(dot_products), dot_products.shape)

        return dot_products == dot_products[i, j]

    def zlayer_to_intercepted_voxel(
        self,
        k_layer_center: int,
        unit_pointing_vec_reference: jnp.ndarray,
        detector_position: jnp.ndarray = None,
    ):
        """
        For a given altitude layer, return the coordinates of the closest intercepted voxel given a pointing direction.
        Parameters
        ----------
        k_layer_center : int
            The k index of the layer in altitude corresponding to k_layer_center.
        unit_pointing_vec_reference : jnp.ndarray
            A unit vector of shape (3,) representing the reference pointing direction.
        detector_position : jnp.ndarray, optional
            The position of the detector in meters. Default is [0.0, 0.0, 0.0].
        Returns
        -------
        mask: jnp.ndarray
            A boolean array indicating which voxels are within the angular tolerance.
        """

        i, j = jnp.meshgrid(
            jnp.arange(self.N), jnp.arange(self.N), indexing="ij"
        )  # shape (N,N)
        k = jnp.full_like(i, k_layer_center)  # shape (N,N)

        indices_2d = jnp.stack([i, j, k], axis=-1)  # shape (N,N,3)

        unit_pointing_vecs = self.pointing_vec_center_voxel(
            indices_grid=indices_2d, detector_position=detector_position
        )  # shape (N,N,3)

        voxels_within_angle = self.intercepted_voxel_with_pointing_detector(
            unit_pointing_vecs, unit_pointing_vec_reference
        )  # shape (N,N)

        masked_indices = jnp.where(voxels_within_angle[..., None], indices_2d, 0)

        return masked_indices

    def zslice_to_intercepted_voxel(
        self,
        unit_pointing_vec_reference: jnp.ndarray,
        kmin: int = None,
        kmax: int = None,
        detector_position: jnp.ndarray = None,
    ):
        """
        For a given altitude slice, return the coordinates of the intercepted voxels given a pointing direction.
        Parameters
        ----------
        kmin, kmax : int
            The k indices defining the slice in altitude. Nslice = kmax - kmin.
        unit_pointing_vec_reference : jnp.ndarray
            A unit vector of shape (3,) representing the reference pointing direction.
        detector_position : jnp.ndarray, optional
            The position of the detector in meters. Default is [0.0, 0.0, 0.0].
        Returns
        -------
        mask: jnp.ndarray
            A boolean array indicating which voxels are within the angular tolerance.
        """
        if kmin is None:
            kmin = detector_position[-1] // self.grid_spacing
        if kmax is None:
            kmax = self.end

        k_slice = jnp.arange(kmin, kmax)

        def map_func(k):
            return self.zlayer_to_intercepted_voxel(
                k_layer_center=k,
                unit_pointing_vec_reference=unit_pointing_vec_reference,
                detector_position=detector_position,
            )

        voxels_indices_slices = jax.vmap(map_func)(k_slice)

        voxels_indices_slices = jnp.swapaxes(voxels_indices_slices, axis1=0, axis2=-1)

        return voxels_indices_slices  # shape (3,N,N,Nslice)

    def zlice_to_selected_voxels_along_los(
        self,
        unit_pointing_vec_reference: jnp.ndarray,
        kmin: int = None,
        kmax: int = None,
        detector_position: jnp.ndarray = None,
        max_radius: float = None,
    ):
        """
        For a given altitude slice, return the coordinates of the selected voxels along the line of sight defined by a pointing direction.
        Parameters
        ----------
        kmin, kmax : int
            The k indices defining the slice in altitude. Nslice = kmax - kmin.
        unit_pointing_vec_reference : jnp.ndarray
            A unit vector of shape (3,) representing the reference pointing direction.
        detector_position : jnp.ndarray, optional
            The position of the detector in meters. Default is [0.0, 0.0, 0.0].
        max_radius : float, optional
            Maximum radius from the detector position to consider voxels. Default is None.
        Returns
        -------
        selected_voxels: jnp.ndarray
            Array of shape (3,N_selected) containing the coordinates of the nearest voxel to the pointing direction for each slice within max_radius.
        """

        max_radius = (
            self.N * self.grid_spacing - detector_position[2]
            if max_radius is None
            else max_radius
        )

        voxels_indices_slices = self.zslice_to_intercepted_voxel(
            unit_pointing_vec_reference=unit_pointing_vec_reference,
            kmin=kmin,
            kmax=kmax,
            detector_position=detector_position,
        )  # shape (3,N,N,Nslice)

        # flatten and keep only non-zero centers
        nonzero_mask = jnp.any(voxels_indices_slices > 0, axis=0)
        # Extract the selected voxel centers
        selected_voxels = voxels_indices_slices[:, nonzero_mask]  # shape (3, Nslice)
        mask_radius = (
            jnp.linalg.norm(
                selected_voxels * self.grid_spacing - detector_position[:, jnp.newaxis],
                axis=0,
            )
            <= max_radius
        )

        return selected_voxels[:, mask_radius]  # shape (3, N_selected)
