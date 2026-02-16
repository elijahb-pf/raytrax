---
icon: lucide/activity
---

The **absorption coefficient** $\alpha$ (units 1/m) is related to the (dimensionless) **optical depth** $\tau$ as

$$\frac{d\tau}{ds}=\alpha$$

and the **linear absorption power density** (units W/m) along the ray tracectory is given as

$$\frac{dP}{ds}=P_0\alpha e^{-\tau}$$

where $P_0$ is the initial power of the ray.

Splitting the dielectric tensor into its Hermitian and anti-Hermitian parts, $\boldsymbol{\varepsilon} = \boldsymbol{\varepsilon}_H + i \boldsymbol{\varepsilon}_A$, the volumetric absorption power density (units W/m³) can be calculated as

$$Q=\frac{\omega \epsilon_0 |E|^2}{2} (\boldsymbol{e}^* \cdot \boldsymbol{\varepsilon}_A \cdot \boldsymbol{e})$$

where $\boldsymbol{E}=E \boldsymbol{e}$ is the local electric field amplitude of the wave with unit polarization vector $\boldsymbol{e}$.

Using further the power flux density

$$\boldsymbol{S}=\ldots$$

the absorption coefficient can be expressed as

$$\alpha = \frac{Q}{|\boldsymbol{S}|}$$