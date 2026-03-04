"""Main module for Raytrax."""

import jax

jax.config.update("jax_enable_x64", True)

from .api import trace as trace
from .equilibrium.interpolate import MagneticConfiguration as MagneticConfiguration
from .types import Beam as Beam
from .types import Interpolators as Interpolators
from .types import RadialProfiles as RadialProfiles
