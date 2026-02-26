"""Main functions to interact with Raytrax."""

import jax
import jax.numpy as jnp
import jaxtyping as jt

from .equilibrium.interpolate import (
    build_magnetic_field_interpolator,
    build_rho_interpolator,
    build_electron_density_profile_interpolator,
    build_electron_temperature_profile_interpolator,
    MagneticConfiguration,
)
from .ray import RaySetting
from .solver import trace_jitted
from .types import (
    Beam,
    BeamProfile,
    Interpolators,
    RadialProfile,
    RadialProfiles,
    TraceBuffers,
    TraceResult,
)


def _bin_power_deposition(
    rho_grid: jt.Float[jax.Array, " nrho"],
    dvolume_drho: jt.Float[jax.Array, " nrho"],
    rho_trajectory: jt.Float[jax.Array, " n"],
    arc_length: jt.Float[jax.Array, " n"],
    linear_power_density: jt.Float[jax.Array, " n"],
) -> jt.Float[jax.Array, " nrho"]:
    """Compute volumetric power deposition profile from ray trajectory.

    Implements δP_abs/δV = Σ_i (dP/ds)_i * δs_i / δV_bin, summing over all
    trajectory segments i that cross the flux-surface shell at rho.
    """
    # Upsample to a fine arc-length grid so each sub-segment spans << one rho bin,
    # giving a smooth histogram without the staircase artifact.
    n_fine = 50 * len(rho_grid)
    s_fine = jnp.linspace(arc_length[0], arc_length[-1], n_fine)
    rho_fine = jnp.interp(s_fine, arc_length, rho_trajectory)
    dpds_fine = jnp.interp(s_fine, arc_length, linear_power_density)

    ds_fine = (arc_length[-1] - arc_length[0]) / (n_fine - 1)
    dP_fine = dpds_fine[:-1] * ds_fine
    rho_mid_fine = 0.5 * (rho_fine[:-1] + rho_fine[1:])

    edges = jnp.concatenate(
        [rho_grid[:1], 0.5 * (rho_grid[:-1] + rho_grid[1:]), rho_grid[-1:]]
    )
    indices = jnp.clip(jnp.searchsorted(edges, rho_mid_fine) - 1, 0, len(rho_grid) - 1)
    power_per_bin = jnp.zeros_like(rho_grid).at[indices].add(dP_fine)

    dV = dvolume_drho * jnp.diff(edges)
    return power_per_bin / dV


def _run_trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
) -> tuple[TraceBuffers, jax.Array]:
    """Build interpolators and run the JIT-compiled ODE solve."""
    setting = RaySetting(frequency=beam.frequency, mode=beam.mode)
    interpolators = Interpolators(
        magnetic_field=build_magnetic_field_interpolator(magnetic_configuration),
        rho=build_rho_interpolator(magnetic_configuration),
        electron_density=build_electron_density_profile_interpolator(radial_profiles),
        electron_temperature=build_electron_temperature_profile_interpolator(
            radial_profiles
        ),
        is_axisymmetric=magnetic_configuration.is_axisymmetric,
    )
    return trace_jitted(
        jnp.asarray(beam.position),
        jnp.asarray(beam.direction),
        setting,
        interpolators,
        magnetic_configuration.nfp,
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
    )


def trace(
    magnetic_configuration: MagneticConfiguration,
    radial_profiles: RadialProfiles,
    beam: Beam,
    trim: bool = True,
) -> TraceResult:
    """Trace a single beam through the plasma.

    Args:
        magnetic_configuration: Magnetic configuration with gridded data
        radial_profiles: Radial profiles of plasma parameters
        beam: Beam initial conditions (position, direction, frequency, mode)
        trim: If True (default), trim the output to the valid trajectory length.
            Set trim=False for gradient-based optimization. The returned
            BeamProfile then contains padded arrays (4097 slots) that are fully
            differentiable w.r.t. beam.position and beam.direction. Padded
            entries have linear_power_density=0 and optical_depth equal to the
            final value, so loss functions like::

                jnp.max(result.beam_profile.optical_depth)
                jnp.sum(result.beam_profile.linear_power_density * weights)

            give the correct answer without trimming. radial_profile is None
            when trim=False.

    Returns:
        TraceResult with beam profile and (if trim=True) radial deposition
        profile.
    """
    result, num_accepted_steps = _run_trace(
        magnetic_configuration, radial_profiles, beam
    )

    if not trim:
        beam_profile = BeamProfile(
            position=result.ode_state[:, :3],
            arc_length=result.arc_length,
            refractive_index=result.ode_state[:, 3:6],
            optical_depth=result.ode_state[:, 6],
            absorption_coefficient=result.absorption_coefficient,
            electron_density=result.electron_density,
            electron_temperature=result.electron_temperature,
            magnetic_field=result.magnetic_field,
            normalized_effective_radius=result.normalized_effective_radius,
            linear_power_density=result.linear_power_density,
        )
        return TraceResult(beam_profile=beam_profile, radial_profile=None)

    # Slot 0 is the antenna position (SaveAt t0=True); accepted steps follow.
    n = num_accepted_steps.item() + 1

    beam_profile = BeamProfile(
        position=result.ode_state[:n, :3],
        arc_length=result.arc_length[:n],
        refractive_index=result.ode_state[:n, 3:6],
        optical_depth=result.ode_state[:n, 6],
        absorption_coefficient=result.absorption_coefficient[:n],
        electron_density=result.electron_density[:n],
        electron_temperature=result.electron_temperature[:n],
        magnetic_field=result.magnetic_field[:n],
        normalized_effective_radius=result.normalized_effective_radius[:n],
        linear_power_density=result.linear_power_density[:n],
    )

    power_binned = _bin_power_deposition(
        magnetic_configuration.rho_1d,
        magnetic_configuration.dvolume_drho,
        result.normalized_effective_radius[:n],
        result.arc_length[:n],
        result.linear_power_density[:n],
    )

    radial_profile = RadialProfile(
        rho=magnetic_configuration.rho_1d,
        volumetric_power_density=power_binned,
    )
    return TraceResult(beam_profile=beam_profile, radial_profile=radial_profile)
