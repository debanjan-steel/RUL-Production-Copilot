# NASA C-MAPSS Sensor Telemetry Glossary

This document details the 21 physical sensors and 3 operational settings measured in the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

---

## Operational Settings

| Setting | Description | Units | Role in FD001 |
| :--- | :--- | :--- | :--- |
| **setting_1** | Altitude | Flight Level / ft | Constant sea-level condition in FD001 (0.0 ± 0.002) |
| **setting_2** | Mach Number | Mach | Constant sea-level condition in FD001 (0.0 ± 0.0005) |
| **setting_3** | Throttle Resolver Angle (TRA) | Degrees | 100.0 (Sea-level takeoff/cruise baseline) |

---

## 21 Turbofan Telemetry Sensors

| Sensor ID | Symbol | Description | Engineering Units | Variance Status in FD001 | Primary Degradation Sensitivity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **s1** | T2 | Total temperature at fan inlet | °R | **Dropped (Zero variance)** | Ambient boundary condition |
| **s2** | T24 | Total temperature at LPC outlet | °R | **Retained (Informative)** | HPC fouling & aerodynamic backpressure |
| **s3** | T30 | Total temperature at HPC outlet | °R | **Retained (Informative)** | HPC efficiency loss, compressor work buildup |
| **s4** | T50 | Total temperature at LPT outlet | °R | **Retained (Informative)** | HPT/LPT wear, EGT margin erosion |
| **s5** | P2 | Pressure at fan inlet | psia | **Dropped (Zero variance)** | Ambient boundary condition |
| **s6** | P15 | Total pressure in bypass duct | psia | **Dropped (Near-zero variance)** | Bypass stream pressure |
| **s7** | P30 | Total pressure at HPC outlet | psia | **Retained (Informative)** | Combustor inlet pressure, compressor delivery |
| **s8** | Nf | Physical fan speed | rpm | **Retained (Informative)** | Low-spool mechanical speed |
| **s9** | Nc | Physical core speed | rpm | **Retained (Informative)** | High-spool mechanical speed, HPC shaft |
| **s10** | Ps30 | Engine pressure ratio (P50/P2) / Static Ps30 | psia | **Dropped (Zero variance in s10 column)** | Reference static indicator |
| **s11** | Ps30 | Static pressure at HPC outlet | psia | **Retained (Informative)** | HPC exit aerodynamics, sensitive to throttling |
| **s12** | phi | Ratio of fuel flow to Ps30 | pps/psi | **Retained (Informative)** | Combustion fuel-to-pressure delivery efficiency |
| **s13** | NRf | Corrected fan speed | rpm | **Retained (Informative)** | Corrected aerodynamic rotor performance |
| **s14** | NRc | Corrected core speed | rpm | **Retained (Informative)** | High-pressure rotor thermodynamic performance |
| **s15** | BPR | Bypass Ratio | - | **Retained (Informative)** | Split between core and bypass air streams |
| **s16** | farB | Burner fuel-air ratio | - | **Dropped (Zero variance)** | Stoichiometric constant in FD001 |
| **s17** | htBleed | Bleed Enthalpy | - | **Retained (Informative)** | HPC bleed extraction enthalpy |
| **s18** | Nf_dmd | Demanded fan speed | rpm | **Dropped (Zero variance)** | FADEC target setpoint |
| **s19** | PCNfR_dmd | Demanded corrected fan speed | rpm | **Dropped (Zero variance)** | FADEC target setpoint |
| **s20** | W31 | HPT coolant bleed flow | lbm/s | **Retained (Informative)** | Turbine clearance & thermal barrier cooling flow |
| **s21** | W32 | LPT coolant bleed flow | lbm/s | **Retained (Informative)** | Low-pressure cooling circuit delivery |

---

## Feature Engineering Summary for FD001

* **Total Sensors Measured:** 21
* **Sensors Dropped Due to Zero/Constant Variance:** 7 sensors (`s1`, `s5`, `s6`, `s10`, `s16`, `s18`, `s19`)
* **Retained Active Sensors:** 14 sensors (`s2`, `s3`, `s4`, `s7`, `s8`, `s9`, `s11`, `s12`, `s13`, `s14`, `s15`, `s17`, `s20`, `s21`)
* **Feature Dimension per Time Step:** 14 features
