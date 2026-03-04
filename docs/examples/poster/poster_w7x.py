"""
W7-X ECRH poster example
=========================

High-density, high-temperature W7-X scenario designed for poster-quality
visualisation of 140 GHz ECRH ray tracing.

The electron density is set close to the O-mode cut-off
(:math:`n_\\mathrm{cut} \\approx 2.43 \\times 10^{20}\\ \\mathrm{m}^{-3}`)
to produce **strongly curved ray paths**, while the electron temperature
ensures **> 80 % single-pass absorption** at the 2nd harmonic of the
electron-cyclotron frequency.

Four figures are produced:

1. **2-D cross-section** — :math:`n_e(R,Z)` background, flux-surface contours,
   2nd-harmonic resonance layer, and beam trajectory coloured by linear power
   density.
2. **Physics along the ray** — :math:`n_e`, :math:`T_e`, :math:`|B|`, and
   :math:`\\rho` as functions of arc length.
3. **Absorption along the ray** — absorption coefficient :math:`\\alpha(s)` and
   cumulative optical depth :math:`\\tau(s)`.
4. **Radial deposition** — volumetric ECRH power density :math:`\\mathrm{d}P/\\mathrm{d}V(\\rho)`.
5. **3-D PyVista visualisation** — last closed flux surface and inner
   :math:`\\rho = 0.5` surface with the beam tube coloured by power density.
"""

# %%
# ## Setup
#
# Suppress deprecation noise, configure poster-quality matplotlib defaults,
# and load the bundled W7-X equilibrium.

import contextlib
import io
import warnings

import jax

jax.config.update("jax_enable_x64", True)

warnings.filterwarnings("ignore", category=DeprecationWarning, module="equinox")
warnings.filterwarnings("ignore", category=UserWarning, message=".*non-interactive.*")

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from raytrax import Beam, RadialProfiles, trace
from raytrax.examples.w7x import (
    PortA,
    get_w7x_magnetic_configuration,
    w7x_aiming_angles_to_direction,
)
from raytrax.plot.plot2d import (
    interpolate_rz_slice,
    plot_beamtrace_rz,
    plot_effective_radius_rz,
)

# Poster-quality matplotlib settings
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

with contextlib.redirect_stdout(io.StringIO()):
    mag_conf = get_w7x_magnetic_configuration()

phi = np.deg2rad(PortA.D1.phi_deg)

# %%
# ## Plasma profiles
#
# The density peak is set to :math:`2.0 \times 10^{20}\ \mathrm{m}^{-3}` —
# close to the O-mode cut-off — so refraction curves the ray strongly.
# The temperature peak at 3.5 keV ensures single-pass absorption well above
# 80 %.  Both profiles are parabolic in :math:`\rho`.

rho_prof = jnp.linspace(0, 1, 500)

# Peak density just below the 140 GHz O-mode cut-off (≈ 2.43 × 10^20 m^-3)
# → strong index-of-refraction gradient → strongly curved ray
ne_peak = 2.0  # 10^20 m^-3
te_peak = 3.5  # keV

profiles = RadialProfiles(
    rho=rho_prof,
    electron_density=ne_peak * (1.0 - rho_prof**2),
    electron_temperature=te_peak * (1.0 - rho_prof**2),
)

# %%
# ## Beam definition and ray tracing
#
# Port-A D1 launcher, downward poloidal aim
# (:math:`\theta_\mathrm{pol} = -20°`) to maximize the path length through
# the plasma core and show pronounced curvature.

beam = Beam(
    position=jnp.array(PortA.D1.cartesian),
    direction=jnp.array(
        w7x_aiming_angles_to_direction(
            theta_pol_deg=-20.0,
            theta_tor_deg=0.0,
            antenna_phi_deg=PortA.D1.phi_deg,
        )
    ),
    frequency=jnp.array(140e9),
    mode="O",
    power=1e6,  # 1 MW
)

result = trace(mag_conf, profiles, beam)

tau_final = float(result.beam_profile.optical_depth[-1])
absorbed = float(result.absorbed_power_fraction)
rho_mean = float(result.deposition_rho_mean)

print(f"Optical depth   τ = {tau_final:.3f}")
print(
    f"Absorbed power    = {absorbed:.1%}  ({result.absorbed_power / 1e3:.0f} kW of 1 MW)"
)
print(f"Deposition ⟨ρ⟩  = {rho_mean:.3f}  ± {float(result.deposition_rho_std):.3f}")

# %%
# ## Figure 1: 2-D poloidal cross-section
#
# The background shows :math:`n_e(R,Z)`.  Thin grey lines are flux-surface
# contours.  The dashed red contour marks the 2nd-harmonic EC resonance at
# :math:`B_\mathrm{res} = f / (2\,f_\mathrm{ce}/\mathrm{T}) \approx 2.502\ \mathrm{T}`.
# The beam is coloured by linear power density.

# Pre-compute the R-Z slice (used for both the ne background and resonance line)
slice_data = interpolate_rz_slice(mag_conf, phi, n_r=300, n_z=300)

# Interpolate ne onto the R-Z grid
ne_2d = np.interp(
    slice_data.rho.ravel(),
    np.array(profiles.rho),
    np.array(profiles.electron_density),
    left=0.0,
    right=0.0,
).reshape(slice_data.rho.shape)
ne_2d = np.where(slice_data.rho <= 1.0, ne_2d, np.nan)

# 2nd-harmonic EC resonance: B_res = f_0 / (2 × 27.99 GHz/T)
B_res = 140e9 / (2.0 * 27.99e9)  # ≈ 2.502 T

fig, ax = plt.subplots(figsize=(5.5, 7))

# ne filled contour background
cont = ax.contourf(
    slice_data.R,
    slice_data.Z,
    ne_2d,
    levels=20,
    cmap="Blues",
    vmin=0,
)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(cont, cax=cax, label=r"$n_e\ [10^{20}\ \mathrm{m}^{-3}]$")

# Flux-surface contours
plot_effective_radius_rz(
    mag_conf, phi=phi, ax=ax, colors="white", linewidths=0.8, alpha=0.7
)

# 2nd-harmonic resonance layer (dashed red)
ax.contour(
    slice_data.R,
    slice_data.Z,
    slice_data.B,
    levels=[B_res],
    colors=["crimson"],
    linewidths=2.0,
    linestyles="--",
)
# Dummy artist for legend
ax.plot(
    [],
    [],
    color="crimson",
    lw=2,
    ls="--",
    label=rf"$B = {B_res:.3f}\ \mathrm{{T}}$ (2nd harm.)",
)

# Beam trace coloured by linear power density
plot_beamtrace_rz(result.beam_profile, phi=phi, ax=ax, lw=3, add_colorbar=False)

# Power-density colorbar alongside beam trace
p_beam = np.array(result.beam_profile.linear_power_density) / 1e6  # MW/m
norm_beam = mcolors.Normalize(vmin=0.0, vmax=float(p_beam.max()))
sm = cm.ScalarMappable(cmap="plasma", norm=norm_beam)
sm.set_array([])
# Attach second colorbar below the first
cax2 = divider.append_axes("right", size="5%", pad=0.6)
plt.colorbar(sm, cax=cax2, label="Linear power density [MW/m]")

ax.legend(loc="upper left", fontsize=10, framealpha=0.8)
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
ax.set_title(
    rf"W7-X 140 GHz O-mode,  $n_{{e,0}} = {ne_peak:.1f}\times10^{{20}}\ \mathrm{{m}}^{{-3}}$,"
    f"\n$T_{{e,0}} = {te_peak:.1f}$ keV,  absorbed = {absorbed:.0%}"
)

plt.tight_layout()
plt.show()

# %%
# ## Figure 2: Physics along the ray
#
# Four panels show electron density, temperature, magnetic-field magnitude,
# and normalised effective radius as the beam propagates along its arc.

s = np.array(result.beam_profile.arc_length)
ne_s = np.array(result.beam_profile.electron_density)
te_s = np.array(result.beam_profile.electron_temperature)
B_s = np.linalg.norm(np.array(result.beam_profile.magnetic_field), axis=-1)
rho_s = np.array(result.beam_profile.normalized_effective_radius)

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)

# ne(s)
axes[0, 0].plot(s, ne_s, color="steelblue")
axes[0, 0].set_ylabel(r"$n_e\ [10^{20}\ \mathrm{m}^{-3}]$")
axes[0, 0].set_title(r"Electron density")

# Te(s)
axes[0, 1].plot(s, te_s, color="darkorange")
axes[0, 1].set_ylabel(r"$T_e\ [\mathrm{keV}]$")
axes[0, 1].set_title(r"Electron temperature")

# |B|(s) with resonance line
axes[1, 0].plot(s, B_s, color="seagreen")
axes[1, 0].axhline(
    B_res,
    color="crimson",
    ls="--",
    lw=1.5,
    label=rf"$B_\mathrm{{res}} = {B_res:.3f}$ T",
)
axes[1, 0].set_xlabel("Arc length [m]")
axes[1, 0].set_ylabel(r"$|B|\ [\mathrm{T}]$")
axes[1, 0].set_title(r"Magnetic field magnitude")
axes[1, 0].legend(fontsize=10)

# rho(s)
axes[1, 1].plot(s, rho_s, color="mediumpurple")
axes[1, 1].axhline(1.0, color="0.5", ls=":", lw=1.5, label="LCFS")
axes[1, 1].set_xlabel("Arc length [m]")
axes[1, 1].set_ylabel(r"$\rho$")
axes[1, 1].set_title(r"Normalised effective radius")
axes[1, 1].legend(fontsize=10)

for ax in axes.flat:
    ax.grid(True, alpha=0.3)

fig.suptitle("Plasma parameters along the ray path", fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# %%
# ## Figure 3: Absorption along the ray
#
# The absorption coefficient :math:`\alpha(s)` (left axis) rises steeply near
# the 2nd-harmonic resonance, while the optical depth :math:`\tau(s)` (right
# axis) accumulates to its final value.

alpha_s = np.array(result.beam_profile.absorption_coefficient)
tau_s = np.array(result.beam_profile.optical_depth)

fig, ax1 = plt.subplots(figsize=(7, 4))

color_alpha = "deeppink"
color_tau = "royalblue"

ax1.plot(s, alpha_s, color=color_alpha, label=r"$\alpha(s)$")
ax1.set_xlabel("Arc length [m]")
ax1.set_ylabel(r"Absorption coefficient $\alpha\ [\mathrm{m}^{-1}]$", color=color_alpha)
ax1.tick_params(axis="y", labelcolor=color_alpha)

ax2 = ax1.twinx()
ax2.plot(s, tau_s, color=color_tau, ls="--", label=r"$\tau(s)$")
ax2.set_ylabel(r"Optical depth $\tau$", color=color_tau)
ax2.tick_params(axis="y", labelcolor=color_tau)
ax2.axhline(tau_final, color=color_tau, ls=":", lw=1, alpha=0.6)
ax2.text(
    s[-1] * 0.98,
    tau_final * 1.03,
    rf"$\tau_\infty = {tau_final:.2f}$",
    ha="right",
    color=color_tau,
    fontsize=11,
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=11)

ax1.set_title(rf"Absorption: $\tau = {tau_final:.2f}$  →  {absorbed:.0%} absorbed")
ax1.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %%
# ## Figure 4: Radial power deposition
#
# `trace` with the default ``trim=True`` provides a `RadialProfile` binned
# onto 50 flux-surface shells.  The peaked deposition near :math:`\rho \approx 0`
# is characteristic of on-axis 2nd-harmonic O-mode heating.
#
# The left panel shows the input profiles; the right panel shows the
# volumetric power density together with the deposition centroid.

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Input profiles
ax = axes[0]
ax.plot(
    np.array(profiles.rho),
    np.array(profiles.electron_density),
    label=r"$n_e\ [10^{20}\ \mathrm{m}^{-3}]$",
)
ax.plot(
    np.array(profiles.rho),
    np.array(profiles.electron_temperature),
    label=r"$T_e\ [\mathrm{keV}]$",
)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("Value")
ax.set_xlim(0, 1)
ax.legend()
ax.set_title("Input plasma profiles")

# Radial power deposition
ax = axes[1]
rho_dep = np.array(result.radial_profile.rho)
power_dep = np.array(result.radial_profile.volumetric_power_density) / 1e6  # MW/m³
ax.fill_between(rho_dep, power_dep, alpha=0.35, color="tomato")
ax.plot(rho_dep, power_dep, color="tomato", label=r"$\mathrm{d}P/\mathrm{d}V$")
ax.axvline(
    rho_mean,
    color="k",
    ls="--",
    lw=1.5,
    label=rf"$\langle\rho\rangle = {rho_mean:.2f}$",
)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel(r"Volumetric power density $[\mathrm{MW/m}^3]$")
ax.set_xlim(0, 1)
ax.legend()
ax.set_title(rf"ECRH power deposition  ({result.absorbed_power / 1e3:.0f} kW)")

plt.tight_layout()
plt.show()

# %%
# ## Figure 5: 3-D PyVista visualisation
#
# PyVista renders the last closed flux surface (semi-transparent steel blue),
# an inner :math:`\rho = 0.5` surface (lighter), and the beam trajectory as a
# tube coloured by linear power density.
#
# The **trame** backend provides a fully interactive WebGL widget directly
# inside the VS Code / JupyterLab notebook cell.
#
# Four objects are overlaid:
#
# * LCFS (:math:`\rho = 1`, steel blue, translucent)
# * Inner surface (:math:`\rho = 0.5`, coral, very translucent)
# * **2nd-harmonic EC resonance surface** (:math:`B_\mathrm{res} \approx 2.502\ \mathrm{T}`,
#   gold) — where the beam deposits its power
# * Beam tube coloured by linear power density (plasma colormap)

import pyvista as pv

from raytrax.plot.plot3d import (
    plot_b_surface_3d,
    plot_beam_profile_3d,
    plot_flux_surface_3d,
)

pv.set_jupyter_backend("trame")
pv.global_theme.background = "white"

plotter = pv.Plotter(notebook=True, window_size=[900, 600])
plotter.add_axes(line_width=3)

# Outer flux surface (LCFS, ρ = 1)
plot_flux_surface_3d(
    mag_conf,
    rho_value=1.0,
    plotter=plotter,
    color="steelblue",
    opacity=0.18,
    smooth_shading=True,
)

# Inner flux surface (ρ = 0.5)
plot_flux_surface_3d(
    mag_conf,
    rho_value=0.5,
    plotter=plotter,
    color="lightcoral",
    opacity=0.15,
    smooth_shading=True,
)

# 2nd-harmonic EC resonance surface: B_res = f_0 / (2 × f_ce/T)
B_res = 140e9 / (2.0 * 27.99e9)  # ≈ 2.502 T
plot_b_surface_3d(
    mag_conf,
    b_value=B_res,
    plotter=plotter,
    color="gold",
    opacity=0.55,
    smooth_shading=True,
)

# Beam tube coloured by linear power density
plot_beam_profile_3d(
    result.beam_profile,
    plotter=plotter,
    tube_radius=0.025,
    n_spline_points=200,
)

plotter.view_isometric()
plotter.camera.zoom(1.2)
plotter.show()

# %%
# ## Export: high-resolution PNG for HTML poster
#
# **How to set the camera angle** (3 steps):
#
# 1. Run the PyVista cell above and rotate/zoom the widget to the angle you want.
# 2. Run this in a new cell::
#
#        print(plotter.camera_position)
#
#    The trame backend syncs the interactive camera back to Python, so this
#    prints the exact view you see in the widget.
# 3. Paste the result below and re-run this cell.

# Paste your camera position here (or keep None to fall back to view_isometric):
camera_position = None  # e.g. [(12.3, -4.1, 3.8), (5.5, 0.0, 0.1), (0.0, 0.0, 1.0)]

export_plotter = pv.Plotter(off_screen=True, window_size=[900, 600])
# No add_axes — axes widget looks odd in poster images

plot_flux_surface_3d(
    mag_conf,
    rho_value=1.0,
    plotter=export_plotter,
    color="steelblue",
    opacity=0.18,
    smooth_shading=True,
)
plot_flux_surface_3d(
    mag_conf,
    rho_value=0.5,
    plotter=export_plotter,
    color="lightcoral",
    opacity=0.15,
    smooth_shading=True,
)
plot_b_surface_3d(
    mag_conf,
    b_value=B_res,
    plotter=export_plotter,
    color="gold",
    opacity=0.55,
    smooth_shading=True,
)
plot_beam_profile_3d(
    result.beam_profile, plotter=export_plotter, tube_radius=0.025, n_spline_points=200
)

if camera_position is not None:
    export_plotter.camera_position = camera_position
else:
    export_plotter.view_isometric()
    export_plotter.camera.zoom(1.2)

export_plotter.enable_anti_aliasing("ssaa")

png_path = "w7x_poster_3d.png"
export_plotter.screenshot(png_path, scale=4)
export_plotter.close()

from PIL import Image

img = Image.open(png_path)
print(
    f"Written: {png_path}  ({img.width} × {img.height} px, {img.size[0] * img.size[1] / 1e6:.1f} MP)"
)
