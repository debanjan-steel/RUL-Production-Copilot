# Turbofan Engine Maintenance Fault Tree & Root Cause Analysis

This knowledge base documents verified failure modes, telemetry symptom signatures, and operational maintenance protocols for turbofan aircraft engines based on the NASA C-MAPSS operational parameters.

---

## 1. High Pressure Compressor (HPC) Degradation & Fouling

### Description
Accumulation of atmospheric contaminants, aerodynamic wear, or tip clearance widening in the High Pressure Compressor stages leading to reduced compression efficiency and thermal buildup.

* **Primary Telemetry Indicators:**
  * **T24 (Total Temperature at LPC Outlet):** Progressive upward drift (> 642.5 °R).
  * **T30 (Total Temperature at HPC Outlet):** Steady increase (> 1585.0 °R) due to reduced isentropic efficiency.
  * **Ps30 (HPC Static Pressure):** Abnormal drop relative to engine core speed.
  * **Nc (Core Speed):** Slight compensatory increase under closed-loop fuel control.

### Severity: High (P2 - Action Required Within 25 Cycles)

### Recommended Maintenance Action
1. Schedule immediate borescope inspection of HPC stages 5 through 9 for blade erosion and deposit accumulation.
2. Execute an engine core water wash procedure during the next turn-around window.
3. If blade tip clearance exceeds 0.045 inches, schedule module replacement.

---

## 2. High Pressure Turbine (HPT) Thermal Distress & Blade Erosion

### Description
Thermal barrier coating (TBC) spallation, oxidation, and creep deformation of the first-stage HPT nozzle guide vanes and rotating blades resulting from sustained high turbine entry temperatures.

* **Primary Telemetry Indicators:**
  * **T50 (LPT Outlet Temperature / EGT):** Significant rise (> 1400.0 °R); Exhaust Gas Temperature (EGT) margin erosion.
  * **W31 (HPT Coolant Bleed Flow):** Deviation from baseline due to seal deterioration.
  * **phi (Ratio of fuel flow to Ps30):** Elevated fuel ratio needed to maintain target thrust.

### Severity: Critical (P1 - Immediate Grounding / Inspection Within 10 Cycles)

### Recommended Maintenance Action
1. Perform high-priority borescope inspection of HPT stage 1 blades and stator vanes for thermal cracking or material loss.
2. Check EGT margin against certified limits. If margin is depleted, restrict engine to derated takeoff power until overhaul.
3. Replace hot-section turbine module if crack length exceeds maintenance manual limit (AMMs 72-41-00).

---

## 3. Low Pressure Turbine (LPT) Efficiency Loss & Blade Tip Rubbing

### Description
Mechanical wear, unbalance, or blade tip shroud rubbing in the Low Pressure Turbine stages reducing energy extraction from the gas path to drive the fan.

* **Primary Telemetry Indicators:**
  * **Nf (Physical Fan Speed):** Drop in rotational speed relative to commanded fuel flow.
  * **BPR (Bypass Ratio):** Abnormal drift (> 8.5) as work split shifts.
  * **T50 (LPT Outlet Temperature):** Increased residual thermal energy entering the exhaust nozzle.

### Severity: Medium (P3 - Advisory / Plan Maintenance Within 50 Cycles)

### Recommended Maintenance Action
1. Inspect LPT blade seals and stage 1-3 rotor blades for foreign object damage (FOD) or rubbing.
2. Perform vibration spectral analysis on bearing compartments 3 and 4 to rule out rotor unbalance.
3. Re-calibrate turbine active clearance control (TACC) valves.

---

## 4. Fuel Metering Unit (FMU) & Hydromechanical Control Drift

### Description
Drift in the electronic engine controller (EEC) or physical wear in the fuel metering valve causing air-fuel mixture imbalance.

* **Primary Telemetry Indicators:**
  * **phi (Fuel Flow to Ps30 Ratio):** Erratic oscillations or uncommanded steps.
  * **W31 / W32 (Coolant Bleeds):** Rapid fluctuations without throttle command changes.

### Severity: High (P2 - Inspect Within 20 Cycles)

### Recommended Maintenance Action
1. Run diagnostic BITE test on Full Authority Digital Engine Control (FADEC) channels A and B.
2. Inspect fuel filters and differential pressure sensors for clogging.
3. Verify fuel metering unit stepper motor torque and feedback resolver alignment.

---

## 5. Sensor Fault / Instrumentation Calibration Drift

### Description
Degradation of thermocouple probes, pressure transducers, or wiring harness leading to anomalous telemetry without true physical component failure.

* **Primary Telemetry Indicators:**
  * Sudden step-discontinuity in single sensor reading with no correlated shift in adjacent stage sensors (e.g. T30 spikes while Ps30 and T24 remain flat).
  * High model uncertainty (MC Dropout variance $\sigma > \tau$).

### Severity: Cautionary (P3 - Sensor Verification Required)

### Recommended Maintenance Action
1. Conduct loop resistance and harness continuity check on affected sensor channel.
2. Replace suspect transducer probe during scheduled overnight ramp check.
3. Re-zero telemetry signal conditioner before returning engine to service.
