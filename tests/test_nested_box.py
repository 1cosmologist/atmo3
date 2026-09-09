import jax.numpy as jnp
import numpy as np
import pytest

from atmo3.box import Box, NestedBox
from atmo3.grid_utils import GridWorkspace, NestedGridWorkspace
from atmo3.observation import Observer


def _spectrum():
    k = jnp.linspace(0.0, 20.0, 512)
    return {"k": k, "pofk": 1.0 / (1.0 + k**2)}


def test_grid_workspace_cell_size_takes_precedence():
    workspace = GridWorkspace(
        N=(8, 6, 4),
        Lbox=(800.0, 600.0, 400.0),
        cell_size=(2.0, 3.0, 4.0),
    )

    np.testing.assert_allclose(workspace.cell_size, (2.0, 3.0, 4.0))
    np.testing.assert_allclose(workspace.Lbox, (16.0, 18.0, 16.0))
    assert workspace.rshape == (8, 6, 4)
    assert workspace.cshape == (8, 6, 3)


def test_nested_cell_sizes_override_lengths_and_refinement_factor():
    workspace = NestedGridWorkspace(
        N=(8, 6, 4),
        nlevels=3,
        Lbox=(999.0, 999.0, 999.0),
        cell_size=((4.0, 3.0, 2.0), (2.0, 1.5, 1.0), (1.0, 0.75, 0.5)),
        refinement_factor=17,
    )

    np.testing.assert_allclose(workspace.level(0).Lbox, (32.0, 18.0, 8.0))
    np.testing.assert_allclose(workspace.level(2).Lbox, (8.0, 4.5, 2.0))
    assert all(
        workspace.band_limits[index][0] == workspace.band_limits[index - 1][1]
        for index in range(1, workspace.nlevels)
    )


def test_one_cell_size_per_level_is_broadcast_to_three_axes():
    workspace = NestedGridWorkspace(
        N=(8, 6, 4), nlevels=3, cell_size=(4.0, 2.0, 1.0)
    )
    np.testing.assert_allclose(workspace.level(1).cell_size, (2.0, 2.0, 2.0))

    with pytest.raises(ValueError, match="one value per level"):
        NestedGridWorkspace(N=(8, 6, 4), nlevels=2, cell_size=(4.0, 2.0, 1.0))


def test_nested_box_generates_disjoint_bands_reproducibly():
    workspace = NestedGridWorkspace(
        N=(8, 6, 4), nlevels=3, cell_size=(4.0, 2.0, 1.0)
    )
    box = NestedBox(
        grid_wsp=workspace,
        spectrum=_spectrum(),
        seed=123,
        nsub=64,
        field_name="water vapor",
        field_unit="kg / m^3",
    )

    first = box.generate_field_fluctuations(time_step=7)
    second = box.generate_field_fluctuations(time_step=7)
    assert len(first) == 3
    assert all(field.shape == (8, 6, 4) for field in first)
    for first_level, second_level in zip(first, second):
        np.testing.assert_allclose(first_level, second_level)
    total_standard_deviation = np.sqrt(
        sum(float(jnp.var(increment)) for increment in box.increments)
    )
    np.testing.assert_allclose(total_standard_deviation, 1.0, rtol=2e-5)

    for level, increment in enumerate(box.increments):
        delta_k = jnp.fft.rfftn(increment)
        outside = ~workspace.band_mask(level)
        assert float(jnp.max(jnp.abs(jnp.where(outside, delta_k, 0.0)))) < 2e-5


def test_box_accepts_rectangular_geometry_and_cell_size():
    box = Box(
        N=(8, 6, 4),
        Lbox=(1000.0, 1000.0, 1000.0),
        cell_size=(2.0, 3.0, 4.0),
        spectrum=_spectrum(),
        nsub=64,
    )
    field = box.generate_field_fluctuations()

    assert field.shape == (8, 6, 4)
    np.testing.assert_allclose(box.grid_wsp.Lbox, (16.0, 18.0, 16.0))


def test_finest_level_containing_uses_global_bounds():
    workspace = NestedGridWorkspace(
        N=(8, 8, 8),
        nlevels=3,
        Lbox=(80.0, 80.0, 40.0),
        refinement_factor=2,
        telescope_position=(40.0, 40.0, 1000.0),
        site_altitude=1000.0,
    )
    points = jnp.asarray(
        [[1.0, 1.0, 1001.0], [25.0, 25.0, 1001.0], [40.0, 40.0, 1001.0]]
    )
    np.testing.assert_array_equal(workspace.finest_level_containing(points), (0, 1, 2))


def test_observer_samples_and_integrates_every_missing_scale_field():
    workspace = NestedGridWorkspace(
        N=(8, 8, 8), nlevels=3, Lbox=(80.0, 80.0, 40.0), refinement_factor=2
    )
    observer = Observer(
        grid_wsp=workspace,
        boresight=jnp.asarray([40.0, 40.0, 0.0]),
        passband={"freq_GHz": jnp.asarray([1.0]), "g_nu": jnp.asarray([1.0])},
        fwhm_arcmin=1.0,
    )
    los_levels = []
    for level in workspace.levels:
        z = level.grid_axis(2)
        x = jnp.full_like(z, 40.0)
        y = jnp.full_like(z, 40.0)
        los_levels.append(
            jnp.stack((x, y, z, z, jnp.ones_like(z)), axis=-1)[None, :, :]
        )
    observer.los_levels = tuple(los_levels)
    observer.los_obj = observer.los_levels
    fields = tuple(
        jnp.full(level.rshape, 10.0 * (index + 1))
        for index, level in enumerate(workspace.levels)
    )

    samples = observer.scan_component(fields)
    assert len(samples) == 3
    for index, values in enumerate(samples):
        np.testing.assert_allclose(values, 10.0 * (index + 1), rtol=2e-6)

    total, contributions = observer.integrate_component(fields)
    expected = tuple(
        10.0 * (index + 1) * float(level.grid_axis(2)[-1])
        for index, level in enumerate(workspace.levels)
    )
    np.testing.assert_allclose(contributions, np.asarray(expected)[:, None], rtol=2e-6)
    np.testing.assert_allclose(total, sum(expected), rtol=2e-6)
