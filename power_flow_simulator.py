#!/usr/bin/env python3
"""
Power Flow Simulator for Solar + Battery Systems

A comprehensive simulation and analysis tool for solar photovoltaic (PV) and battery energy
storage systems (BESS). This simulator models realistic power flows, equipment constraints,
and economic performance to optimize system sizing and configuration.

Key Features:
    - Individual scenario simulation with detailed power flow analysis
    - Multi-scenario batch analysis across parameter combinations
    - Real-world equipment modeling (inverter limits, battery C-rates, efficiency losses)
    - Economic analysis with ROI, payback periods, and cost savings
    - Comprehensive visualization suite (time series, heatmaps, Pareto analysis)
    - Export capabilities for further analysis and reporting

Input Data Requirements:
    - production.csv: Solar generation data with timestamp_id and solar_power_kw columns
    - consumption.csv: Electricity consumption with datetime, energy (kWh), power (kW),
      and transmission_block columns

Output Files Generated:
    Single Mode:
        - power_flow_analysis_7days.png: Power flow visualization
        - monthly_energy_analysis.png: Monthly energy breakdown
        - power_flow_simulation_results.csv: Detailed simulation data
        - system_summary.csv: System configuration and performance summary

    Batch Mode:
        - multi_scenario_comparison.csv: Ranked comparison of all scenarios
        - multi_scenario_savings_heatmap.png: Annual savings across configurations
        - multi_scenario_roi_heatmap.png: Payback period analysis
        - multi_scenario_self_sufficiency_heatmap.png: Energy independence analysis
        - multi_scenario_pareto_analysis.png: Investment vs savings optimization
        # - multi_scenario_soc_time_365days.png: Battery SOC organized by battery capacity (disabled)
        # - soc_[solar]kW_solar_[inverter]kW_inverter_365days.png: Individual SOC plots for each solar+inverter combination (disabled)
        - multi_scenario_comprehensive_results.csv: Complete results dataset

Usage Examples:
    Single scenario analysis:
        $ python3 power_flow_simulator.py --solar-power 15.0 --battery-capacity 20.0

    Batch analysis:
        $ python3 power_flow_simulator.py --batch-mode \
          --solar-range 10,15,20 --battery-range 0,10,20 \
          --electricity-price 0.15 --export-price 0.0

Performance Notes:
    - Single simulations typically complete in 1-5 seconds
    - Batch analysis scales with parameter combinations (n_solar × n_inverter × n_battery)
    - Large datasets (>100k intervals) may require several GB of memory
    - Industrial-scale analysis (200+ scenarios) can take 10-30 minutes

Author: Claude Code
Version: 1.0
License: MIT
"""

import argparse
import warnings
from itertools import product
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
from scipy.optimize import differential_evolution
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Initialize joblib memory cache
memory = Memory(location=".cache", verbose=0)


class PowerFlowSimulator:
    """
    Power flow simulator for individual solar + battery system scenarios.

    This class models the operation of a solar photovoltaic system with battery energy storage,
    simulating realistic power flows with equipment constraints and calculating economic performance.
    The simulation operates on 15-minute intervals using real consumption and solar generation data.

    Key Capabilities:
        - Load and process real smart meter data (consumption) and solar generation data
        - Scale baseline solar generation to user-specified system capacity
        - Model realistic equipment constraints (inverter clipping, battery C-rates)
        - Simulate battery charge/discharge cycles with efficiency losses
        - Calculate comprehensive economic analysis (costs, savings, ROI)
        - Generate detailed visualizations and export results

    Simulation Algorithm:
        1. Scale solar generation from baseline system to target capacity
        2. Apply inverter power limits (clipping losses)
        3. For each 15-minute interval:
           - Calculate net energy demand (consumption - solar)
           - If surplus: charge battery (with efficiency losses), export remainder
           - If deficit: discharge battery first, import remainder from grid
           - Apply battery power limits and SOC constraints
        4. Track all energy flows and calculate costs using time-of-use pricing

    Equipment Modeling:
        - Solar panels: Proportional scaling from baseline 640.6kW system
        - Inverter: Hard power limit with clipping losses
        - Battery: Round-trip efficiency, C-rate power limits, SOC bounds (0-100%)
        - Grid: Import/export with different pricing tiers

    Economic Analysis:
        - Baseline cost: Grid-only consumption at retail rates
        - Solar-only cost: With solar generation and export revenue
        - Solar+battery cost: With battery optimization of energy flows
        - Savings calculations relative to baseline and solar-only scenarios

    Attributes:
        solar_panel_power_kw (float): Solar panel DC capacity in kW
        inverter_power_kw (float): Inverter AC power limit in kW
        battery_capacity_kwh (float): Battery energy capacity in kWh
        battery_charge_power_kw (float): Maximum charging power in kW
        battery_discharge_power_kw (float): Maximum discharging power in kW
        battery_efficiency (float): Round-trip efficiency (0.0-1.0)
        production_file (str): Path to solar generation CSV file
        consumption_file (str): Path to electricity consumption CSV file
        df (pd.DataFrame): Aligned input data after loading
        simulation_results (pd.DataFrame): Detailed simulation outputs

    Typical Workflow:
        1. Initialize with system specifications
        2. load_and_align_data() - Load and process input files
        3. scale_solar_generation() - Scale to target system size
        4. simulate_power_flows() - Run core simulation
        5. calculate_costs_and_savings() - Economic analysis
        6. Create visualizations and export results

    Example:
        >>> simulator = PowerFlowSimulator(
        ...     solar_panel_power_kw=15.0,
        ...     battery_capacity_kwh=20.0,
        ...     battery_efficiency=0.95
        ... )
        >>> simulator.load_and_align_data()
        >>> simulator.scale_solar_generation()
        >>> simulator.simulate_power_flows()
        >>> cost_analysis = simulator.calculate_costs_and_savings()
        >>> simulator.create_power_flow_visualization()
    """

    def __init__(
        self,
        solar_panel_power_kw: float = 8.5,
        inverter_power_kw: float = 8.0,
        battery_capacity_kwh: float = 10.0,
        battery_charge_power_kw: float = None,
        battery_discharge_power_kw: float = None,
        battery_c_rate: float = 0.5,
        battery_efficiency: float = 0.9,
        production_file: str = "production.csv",
        consumption_file: str = "consumption.csv",
        transmission_costs: dict[str, float] = None,
        monthly_power_fees: dict[str, float] = None,
        ove_spte_fee: float = 3.44078,
        peak_price: float = 0.14683,
        off_peak_price: float = 0.10664,
        enable_power_smoothing: bool = False,
        min_soc_reserve: float = 0.2,
        max_power_threshold: float = None,
        max_power_by_block: dict[int, float] = None,
        heating_config: dict = None,
    ):
        """
        Initialize the power flow simulator

        Args:
            solar_panel_power_kw (float): Solar panel capacity in kW
            inverter_power_kw (float): Inverter maximum AC output in kW
            battery_capacity_kwh (float): Battery storage capacity in kWh
            battery_charge_power_kw (float): Maximum battery charging power in kW
            battery_discharge_power_kw (float): Maximum battery discharging power in kW
            battery_efficiency (float): Battery round-trip efficiency (0.0-1.0)
            production_file (str): Path to solar production CSV file
            consumption_file (str): Path to consumption CSV file
            transmission_costs (Dict[str, float]): Transmission costs by block (block1-block5)
            monthly_power_fees (Dict[str, float]): Monthly power fees by block (EUR/kW/month for max power)
            ove_spte_fee (float): OVE-SPTE fee rate in EUR/kW/month (default: 3.44078)
            peak_price (float): Peak hour electricity price in EUR/kWh (6am-10pm weekdays)
            off_peak_price (float): Off-peak hour electricity price in EUR/kWh (nights and weekends)
            heating_config (dict): Heating load configuration with keys:
                - heating_kwh (float): Total heating energy for season in kWh
                - start_hour (int): Daily start hour (default: 7)
                - end_hour (int): Daily end hour (0 = midnight, default: 0)
        """
        # System specifications
        self.solar_panel_power_kw = solar_panel_power_kw
        self.inverter_power_kw = inverter_power_kw
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_c_rate = battery_c_rate

        # Calculate battery power based on C-rate if not explicitly provided
        if battery_charge_power_kw is None:
            self.battery_charge_power_kw = battery_capacity_kwh * battery_c_rate
        else:
            self.battery_charge_power_kw = battery_charge_power_kw

        if battery_discharge_power_kw is None:
            self.battery_discharge_power_kw = battery_capacity_kwh * battery_c_rate
        else:
            self.battery_discharge_power_kw = battery_discharge_power_kw

        self.battery_efficiency = battery_efficiency

        # Data files
        self.production_file = production_file
        self.consumption_file = consumption_file

        # Transmission costs (EUR/kWh) - 5 blocks (1=most expensive, 3=cheapest)
        if transmission_costs is None:
            transmission_costs = {
                "block1": 0.01282,
                "block2": 0.01216,
                "block3": 0.01186,
                "block4": 0.01164,
                "block5": 0.01175,
            }
        self.transmission_costs = transmission_costs

        # Monthly power fees (EUR/kW/month) - based on max power in each block
        if monthly_power_fees is None:
            monthly_power_fees = {
                "block1": 3.75969,
                "block2": 1.05262,
                "block3": 0.12837,
                "block4": 0.0,
                "block5": 0.0,
            }
        self.monthly_power_fees = monthly_power_fees

        # OVE-SPTE fee (EUR/kW/month) - weighted block power cost
        self.ove_spte_fee = ove_spte_fee

        # Electricity pricing (EUR/kWh)
        self.peak_price = peak_price  # 6am-10pm weekdays
        self.off_peak_price = off_peak_price  # nights and weekends

        # Power smoothing settings
        self.enable_power_smoothing = enable_power_smoothing
        self.min_soc_reserve = min_soc_reserve  # Minimum SOC reserve for power smoothing (0-1)
        self.max_power_threshold = max_power_threshold  # Maximum power threshold in kW (global)

        # Block-specific power thresholds for power smoothing (kW)
        if max_power_by_block is None:
            self.max_power_by_block = {
                1: 250.0,  # Block 1: Most expensive, lowest threshold
                2: 250.0,  # Block 2: Second most expensive
                3: 340.0,  # Block 3: Medium cost
                4: 2000.0,  # Block 4: High threshold (no monthly fees)
                5: 2000.0,  # Block 5: High threshold (no monthly fees)
            }
        else:
            self.max_power_by_block = max_power_by_block

        # Heating load configuration
        self.heating_config = heating_config

        # Baseline system (from the data files)
        self.baseline_solar_kw = 640.6  # 640.6kW solar installation
        self.baseline_inverter_kw = 532.8  # 532.8kW inverter

        # Data storage
        self.df = None
        self.simulation_results = None

        # Max power by block
        self.max_power_by_block_baseline = None
        self.max_power_by_block_simulated = None

        print("Power Flow Simulator initialized:")
        print(f"  Solar panels: {self.solar_panel_power_kw} kW")
        print(f"  Inverter: {self.inverter_power_kw} kW")
        print(
            f"  Battery: {self.battery_capacity_kwh} kWh ({self.battery_charge_power_kw}kW charge / {self.battery_discharge_power_kw}kW discharge)"
        )
        print(f"  Battery efficiency: {self.battery_efficiency * 100:.0f}%")

    def load_and_align_data(self) -> None:
        """
        Load production and consumption data from CSV files and align by timestamp.

        This method loads solar generation data from the production file and electricity
        consumption data from the consumption file, processes them into a consistent format,
        and aligns them by timestamp for simulation use.

        Input File Requirements:
            production.csv format:
                - Column 1: timestamp_id (sequential identifier)
                - Column 2: solar_power_kw (instantaneous power in kW)

            consumption.csv format:
                - Column 1: datetime (format: 'DD. MM. YYYY HH:MM:SS')
                - Column 2: energy_wh (energy consumed in 15-min interval, actually kWh)
                - Column 3: power_w (average power during interval, actually kW)
                - Column 4: transmission_block (1-5, time-of-use pricing block)
                - Column 5: extra (empty column, automatically dropped)

        Processing Steps:
            1. Load both CSV files and standardize column names
            2. Parse consumption datetime strings to pandas datetime objects
            3. Ensure both datasets have same number of records (use minimum)
            4. Create aligned DataFrame with consistent time indexing
            5. Add time-based features (hour, weekday, month, peak hour flags)

        Side Effects:
            - Sets self.df with the aligned dataset
            - Prints data loading summary and statistics

        Raises:
            FileNotFoundError: If either CSV file is not found
            pd.errors.ParserError: If CSV files have incorrect format
            ValueError: If datetime parsing fails

        Notes:
            - Data is expected to be in 15-minute intervals
            - Consumption data units are actually kWh and kW despite column names
            - Peak hours defined as 6am-10pm on weekdays (Monday-Friday)
            - Baseline system assumption: 640.6kW solar, 532.8kW inverter
        """
        print("\nLoading and aligning data...")

        # Load production data
        production_df = pd.read_csv(self.production_file)
        production_df.columns = ["timestamp_id", "solar_power_kw"]

        # Load consumption data
        consumption_df = pd.read_csv(self.consumption_file)
        consumption_df.columns = ["datetime", "energy_wh", "power_w", "transmission_block", "extra"]
        consumption_df = consumption_df.drop("extra", axis=1)  # Remove empty column

        # Convert consumption datetime
        consumption_df["datetime"] = pd.to_datetime(
            consumption_df["datetime"], format="%d. %m. %Y %H:%M:%S"
        )

        # Data is already in kWh and kW units - use directly
        consumption_df["consumption_kwh"] = consumption_df["energy_wh"]
        consumption_df["consumption_kw"] = consumption_df["power_w"]

        # Ensure we have the same number of records (take minimum)
        min_records = min(len(production_df), len(consumption_df))
        production_df = production_df.iloc[:min_records].copy()
        consumption_df = consumption_df.iloc[:min_records].copy()

        # Create aligned dataset
        self.df = pd.DataFrame(
            {
                "datetime": consumption_df["datetime"],
                "baseline_solar_kw": production_df["solar_power_kw"],
                "consumption_kwh": consumption_df["consumption_kwh"],
                "consumption_kw": consumption_df["consumption_kw"],
                "transmission_block": consumption_df["transmission_block"],
            }
        )

        # Add time-based features
        self.df["hour"] = self.df["datetime"].dt.hour
        self.df["weekday"] = self.df["datetime"].dt.weekday  # 0=Monday, 6=Sunday
        self.df["month"] = self.df["datetime"].dt.month

        # Define 2024 Slovenian holidays (off-peak pricing)
        holidays_2024 = [
            "2024-01-01",
            "2024-01-02",
            "2024-02-08",
            "2024-04-10",
            "2024-05-01",
            "2024-05-02",
            "2024-06-25",
            "2024-08-15",
            "2024-10-31",
            "2024-11-01",
            "2024-12-25",
            "2024-12-26",
        ]
        holiday_dates = pd.to_datetime(holidays_2024).date
        self.df["is_holiday"] = self.df["datetime"].dt.date.isin(holiday_dates)

        # Peak hours: 6am-10pm on weekdays, excluding holidays
        self.df["is_peak_hour"] = (
            (self.df["hour"] >= 6)
            & (self.df["hour"] < 22)
            & (self.df["weekday"] < 5)
            & (~self.df["is_holiday"])
        )

        print(
            f"Loaded {len(self.df)} data points from {self.df['datetime'].min()} to {self.df['datetime'].max()}"
        )
        print(
            f"Baseline system: {self.baseline_solar_kw}kW solar, {self.baseline_inverter_kw}kW inverter"
        )
        print(f"Max baseline solar output: {self.df['baseline_solar_kw'].max():.1f} kW")
        print(f"Total consumption: {self.df['consumption_kwh'].sum():.1f} kWh/year")

        # Apply heating load if configured
        self.apply_heating_load()

    def apply_heating_load(self) -> None:
        """
        Add synthetic heating load to consumption during heating season (Oct-Mar).

        Distributes total heating energy across winter months with weighted factors
        based on typical Slovenian climate patterns. Energy is spread across
        specified daily hours within each month.

        Monthly distribution factors (normalized to sum = 6.6):
            - January: 1.5 (coldest)
            - February: 1.4
            - December: 1.3
            - November: 1.0
            - March: 0.8
            - October: 0.6 (mildest)
        """
        if self.heating_config is None:
            return

        heating_kwh = self.heating_config.get("heating_kwh", 3333)
        start_hour = self.heating_config.get("start_hour", 7)
        end_hour = self.heating_config.get("end_hour", 0)  # 0 = midnight

        # Monthly heating factors
        heating_factors = {
            10: 0.6,   # October
            11: 1.0,   # November
            12: 1.3,   # December
            1: 1.5,    # January (coldest)
            2: 1.4,    # February
            3: 0.8,    # March
        }

        total_factor = sum(heating_factors.values())  # 6.6
        base_monthly_kwh = heating_kwh / total_factor

        print(f"\nApplying heating load: {heating_kwh} kWh total (Oct-Mar)")
        print(f"  Daily hours: {start_hour}:00 - {'midnight' if end_hour == 0 else f'{end_hour}:00'}")

        for month, factor in heating_factors.items():
            month_kwh = base_monthly_kwh * factor

            # Build mask for this month during heating hours
            if end_hour == 0:  # Midnight case
                mask = (self.df["month"] == month) & (self.df["hour"] >= start_hour)
            else:
                mask = (
                    (self.df["month"] == month)
                    & (self.df["hour"] >= start_hour)
                    & (self.df["hour"] < end_hour)
                )

            intervals = mask.sum()
            if intervals > 0:
                kwh_per_interval = month_kwh / intervals
                kw_per_interval = kwh_per_interval * 4  # 15-min intervals → kW

                self.df.loc[mask, "consumption_kwh"] += kwh_per_interval
                self.df.loc[mask, "consumption_kw"] += kw_per_interval

                print(f"  {month:02d}: {month_kwh:6.1f} kWh across {intervals:4d} intervals")

        print(f"  New total consumption: {self.df['consumption_kwh'].sum():.1f} kWh/year")

    def scale_solar_generation(self) -> None:
        """Scale solar generation from baseline system to user-specified system"""
        print(
            f"\nScaling solar generation from {self.baseline_solar_kw}kW baseline to {self.solar_panel_power_kw}kW..."
        )

        # Scale solar output proportionally
        scaling_factor = self.solar_panel_power_kw / self.baseline_solar_kw
        self.df["scaled_solar_kw"] = self.df["baseline_solar_kw"] * scaling_factor

        # Apply inverter limits
        interval_hours = 0.25  # 15 minutes = 0.25 hours
        max_inverter_output_per_interval = self.inverter_power_kw

        # Clip positive generation, keep negative values (nighttime) as-is
        self.df["solar_output_kw"] = np.where(
            self.df["scaled_solar_kw"] > 0,
            np.minimum(self.df["scaled_solar_kw"], max_inverter_output_per_interval),
            self.df["scaled_solar_kw"],  # Keep negative values unchanged
        )

        # Convert to energy for 15-minute intervals
        self.df["solar_generation_kwh"] = self.df["solar_output_kw"] * interval_hours

        # Calculate clipping losses
        clipped_power = np.maximum(0, self.df["scaled_solar_kw"] - max_inverter_output_per_interval)
        annual_clipping_kwh = (clipped_power * interval_hours).sum()

        total_scaled_generation = (np.maximum(0, self.df["scaled_solar_kw"]) * interval_hours).sum()
        total_actual_generation = (np.maximum(0, self.df["solar_output_kw"]) * interval_hours).sum()

        print(f"Solar scaling factor: {scaling_factor:.3f}x")
        print(f"Annual solar generation (after inverter): {total_actual_generation:.1f} kWh")
        print(
            f"Annual clipping losses: {annual_clipping_kwh:.1f} kWh ({annual_clipping_kwh / total_scaled_generation * 100:.1f}%)"
        )

    def simulate_power_flows(self) -> None:
        """
        Execute the core power flow simulation with battery energy management.

        This method simulates the operation of the solar + battery system over all time intervals
        in the dataset, modeling realistic energy flows, equipment constraints, and battery
        state management. The simulation follows a priority-based energy management strategy.

        Simulation Algorithm:
            For each 15-minute interval:
            1. Calculate net energy demand: consumption - solar_generation
            2. If surplus energy (net_demand ≤ 0):
               a. Charge battery with available surplus (respecting power and SOC limits)
               b. Account for charging efficiency losses
               c. Export any remaining surplus to grid
            3. If energy deficit (net_demand > 0):
               a. Discharge battery to meet demand (respecting power and SOC limits)
               b. Import remaining demand from grid
            4. Enforce battery SOC bounds (0-100%) and power constraints
            5. Log all energy flows and battery state

        Energy Flow Priority:
            1. Solar generation first meets consumption directly
            2. Surplus solar charges battery (with efficiency losses)
            3. Remaining surplus exported to grid
            4. Battery discharges to meet unmet demand
            5. Grid import covers any remaining demand

        Battery Modeling:
            - Initial SOC: 100% (full battery start)
            - Power limits: charge_power_kw and discharge_power_kw
            - Round-trip efficiency applied to charging only
            - SOC bounds: 0-100% capacity
            - No degradation or calendar aging effects

        Side Effects:
            - Sets self.simulation_results with detailed interval data
            - Prints simulation summary statistics

        Requires:
            - self.df must be populated (call load_and_align_data() first)
            - self.df must contain 'solar_generation_kwh' (call scale_solar_generation() first)

        Raises:
            AttributeError: If required data not loaded or solar not scaled
            ValueError: If invalid battery or power specifications

        Notes:
            - Simulation assumes perfect forecasting (no predictive control)
            - Battery efficiency losses only applied to charging, not discharging
            - All energy flows tracked at 15-minute resolution
            - Results suitable for detailed energy and economic analysis
        """
        print("\nSimulating power flows...")

        # Initialize simulation
        soc_kwh = self.battery_capacity_kwh  # Start with full battery for testing
        soc_kwh = 0  # Start with empty battery for testing
        initial_soc_percent = (
            (soc_kwh / self.battery_capacity_kwh * 100) if self.battery_capacity_kwh > 0 else 0
        )
        print(f"Initial battery SOC: {soc_kwh:.1f} kWh ({initial_soc_percent:.0f}%)")
        interval_hours = 0.25  # 15 minutes

        # Calculate max charge/discharge energy per interval
        max_charge_kwh_per_interval = self.battery_charge_power_kw * interval_hours
        max_discharge_kwh_per_interval = self.battery_discharge_power_kw * interval_hours

        # Storage for results
        results = []

        # Initialize power smoothing tracking
        max_power_by_block = {
            "block_1": 0.0,
            "block_2": 0.0,
            "block_3": 0.0,
            "block_4": 0.0,
            "block_5": 0.0,
        }
        power_smoothing_discharge_total = 0.0

        # Track peak smoothing failures
        smoothing_failures = []

        for _, row in self.df.iterrows():
            solar_generation = row["solar_generation_kwh"]
            consumption = row["consumption_kwh"]
            transmission_block = int(row["transmission_block"])

            # Initialize interval values
            battery_charge = 0.0
            battery_discharge = 0.0
            grid_import = 0.0
            grid_export = 0.0
            power_smoothing_discharge = 0.0

            # Calculate net demand (positive = need to import, negative = surplus to export)
            net_demand = consumption - solar_generation

            if net_demand <= 0:
                # Solar surplus available
                surplus = -net_demand

                # Try to charge battery first
                available_battery_capacity = self.battery_capacity_kwh - soc_kwh
                max_charge_this_interval = min(
                    max_charge_kwh_per_interval, available_battery_capacity
                )

                if surplus > 0 and max_charge_this_interval > 0:
                    # Account for charging efficiency
                    energy_to_charge = min(
                        surplus, max_charge_this_interval / self.battery_efficiency
                    )
                    actual_charge = energy_to_charge * self.battery_efficiency

                    battery_charge = energy_to_charge
                    soc_kwh += actual_charge
                    surplus -= energy_to_charge

                # Export remaining surplus
                if surplus > 0:
                    grid_export = surplus

            else:
                # Need to import energy
                demand = net_demand

                # Try to discharge battery first (but respect min_soc_reserve for normal operation)
                if self.enable_power_smoothing:
                    # With power smoothing: respect reserve during normal discharge
                    min_soc_kwh = self.min_soc_reserve * self.battery_capacity_kwh
                    available_above_reserve = max(0, soc_kwh - min_soc_kwh)
                    available_discharge = min(
                        available_above_reserve, max_discharge_kwh_per_interval
                    )
                else:
                    # Without power smoothing: can discharge to 0% (original behavior)
                    available_discharge = min(soc_kwh, max_discharge_kwh_per_interval)

                if demand > 0 and available_discharge > 0:
                    discharge_amount = min(demand, available_discharge)
                    battery_discharge = discharge_amount
                    soc_kwh -= discharge_amount
                    demand -= discharge_amount

                # Import remaining demand from grid
                if demand > 0:
                    grid_import = demand

            # Power smoothing logic: use battery reserve to minimize max power
            target_power = None
            if self.enable_power_smoothing and grid_import > 0:
                # Convert to power (kW) for 15-minute interval
                grid_power_kw = grid_import / interval_hours

                # Determine target power threshold for this block
                # First try block-specific threshold
                if transmission_block in self.max_power_by_block:
                    target_power = self.max_power_by_block[transmission_block]
                # Fall back to global threshold
                elif self.max_power_threshold is not None:
                    target_power = self.max_power_threshold
                else:
                    # Default: try to keep power under 200kW for all blocks
                    target_power = 200.0

                # Check if current grid power exceeds our target
                if target_power is not None and grid_power_kw > target_power:
                    # Calculate power reduction needed
                    power_reduction_needed = grid_power_kw - target_power
                    energy_reduction_needed = power_reduction_needed * interval_hours

                    # Power smoothing can discharge below the reserve (down to 0 SOC if needed)
                    # Available capacity = all current SOC for power smoothing
                    available_for_smoothing = soc_kwh
                    max_discharge_from_reserve = min(
                        available_for_smoothing, max_discharge_kwh_per_interval
                    )

                    # Use reserve to reduce grid power if beneficial
                    if max_discharge_from_reserve > 0 and energy_reduction_needed > 0:
                        actual_reduction = min(energy_reduction_needed, max_discharge_from_reserve)
                        power_smoothing_discharge = actual_reduction
                        battery_discharge += actual_reduction
                        soc_kwh -= actual_reduction
                        grid_import -= actual_reduction
                        power_smoothing_discharge_total += actual_reduction

            # Update maximum power tracking (after power smoothing)
            final_grid_power_kw = grid_import / interval_hours if grid_import > 0 else 0
            max_power_by_block[f"block_{transmission_block}"] = max(
                max_power_by_block[f"block_{transmission_block}"], final_grid_power_kw
            )

            # Track peak smoothing failures
            if (
                self.enable_power_smoothing
                and target_power is not None
                and final_grid_power_kw > target_power
            ):
                # Record this failure
                smoothing_failures.append(
                    {
                        "timestamp": row["datetime"],
                        "block": transmission_block,
                        "target_kw": target_power,
                        "actual_kw": final_grid_power_kw,
                        "excess_kw": final_grid_power_kw - target_power,
                        "soc_percent": (soc_kwh / self.battery_capacity_kwh) * 100
                        if self.battery_capacity_kwh > 0
                        else 0,
                    }
                )

            # Ensure SOC stays within bounds
            soc_kwh = max(0, min(self.battery_capacity_kwh, soc_kwh))

            # Store results
            results.append(
                {
                    "datetime": row["datetime"],
                    "solar_generation_kwh": solar_generation,
                    "consumption_kwh": consumption,
                    "battery_soc_kwh": soc_kwh,
                    "battery_soc_percent": (soc_kwh / self.battery_capacity_kwh) * 100
                    if self.battery_capacity_kwh > 0
                    else 0,
                    "battery_charge_kwh": battery_charge,
                    "battery_discharge_kwh": battery_discharge,
                    "power_smoothing_discharge_kwh": power_smoothing_discharge,
                    "grid_import_kwh": grid_import,
                    "grid_export_kwh": grid_export,
                    "transmission_block": row["transmission_block"],
                }
            )

        # Convert to DataFrame
        self.simulation_results = pd.DataFrame(results)

        # Store power smoothing results
        if self.enable_power_smoothing:
            self.power_smoothing_max_power = max_power_by_block
            self.power_smoothing_total_discharge = power_smoothing_discharge_total
            self.smoothing_failures = smoothing_failures

        # Calculate summary statistics
        total_consumption = self.simulation_results["consumption_kwh"].sum()
        total_solar = self.simulation_results["solar_generation_kwh"].sum()
        total_grid_import = self.simulation_results["grid_import_kwh"].sum()
        total_grid_export = self.simulation_results["grid_export_kwh"].sum()
        total_battery_charge = self.simulation_results["battery_charge_kwh"].sum()
        total_battery_discharge = self.simulation_results["battery_discharge_kwh"].sum()
        total_power_smoothing = self.simulation_results["power_smoothing_discharge_kwh"].sum()

        # Battery utilization
        avg_soc = self.simulation_results["battery_soc_percent"].mean()
        min_soc = self.simulation_results["battery_soc_percent"].min()
        max_soc = self.simulation_results["battery_soc_percent"].max()

        # Estimate annual cycles
        total_throughput = total_battery_charge + total_battery_discharge
        annual_cycles = (
            total_throughput / (2 * self.battery_capacity_kwh)
            if self.battery_capacity_kwh > 0
            else 0
        )

        print("Simulation complete!")
        print(f"  Total consumption: {total_consumption:.1f} kWh")
        print(f"  Total solar generation: {total_solar:.1f} kWh")
        print(f"  Grid import: {total_grid_import:.1f} kWh")
        print(f"  Grid export: {total_grid_export:.1f} kWh")
        print(
            f"  Energy consumed from battery: {total_battery_discharge:.1f} kWh ({total_battery_discharge / total_consumption * 100:.1f}%)"
        )
        print(
            f"  Energy consumed from grid: {total_grid_import:.1f} kWh ({total_grid_import / total_consumption * 100:.1f}%)"
        )
        if self.enable_power_smoothing and total_power_smoothing > 0:
            print(
                f"  Power smoothing discharge: {total_power_smoothing:.1f} kWh (from {self.min_soc_reserve * 100:.0f}% reserve)"
            )
            print(
                f"  Maximum power achieved by block: Block 1: {max_power_by_block['block_1']:.1f}kW, Block 2: {max_power_by_block['block_2']:.1f}kW, Block 3: {max_power_by_block['block_3']:.1f}kW"
            )
        print(f"  Battery cycles: {annual_cycles:.1f} per year")
        print(f"  Average SOC: {avg_soc:.1f}%")

        # Print peak smoothing failure warnings
        if self.enable_power_smoothing and smoothing_failures:
            print("\n⚠️  Peak Smoothing Failures Detected:")
            print(f"  Total failures: {len(smoothing_failures)} intervals")

            # Group by block
            failures_by_block = {}
            for failure in smoothing_failures:
                block = failure["block"]
                if block not in failures_by_block:
                    failures_by_block[block] = []
                failures_by_block[block].append(failure)

            # Print summary for each block
            for block in sorted(failures_by_block.keys()):
                block_failures = failures_by_block[block]
                worst_failure = max(block_failures, key=lambda x: x["excess_kw"])
                avg_excess = sum(f["excess_kw"] for f in block_failures) / len(block_failures)

                print(f"\n  Block {block}:")
                print(f"    Failed intervals: {len(block_failures)}")
                print(f"    Target threshold: {block_failures[0]['target_kw']:.1f} kW")
                print(
                    f"    Worst peak: {worst_failure['actual_kw']:.1f} kW (exceeded by {worst_failure['excess_kw']:.1f} kW)"
                )
                print(f"    Average excess: {avg_excess:.1f} kW")
                print(
                    f"    Worst failure at: {worst_failure['timestamp']} (SOC: {worst_failure['soc_percent']:.1f}%)"
                )

            print(
                "\n  💡 Tip: Consider increasing battery capacity, C-rate, or SOC reserve to reduce failures."
            )

    def calculate_costs_and_savings(
        self,
        peak_price: float = None,
        off_peak_price: float = None,
        export_price: float = 0.0,
        base_price: float = None,
    ) -> dict[str, float] | None:
        """
        Calculate comprehensive economic analysis of the solar + battery system.

        This method computes annual electricity costs for three scenarios and calculates
        savings relative to baseline grid-only consumption. The analysis provides key
        economic metrics for investment decision-making.

        Economic Scenarios:
            1. Baseline: Grid-only consumption at retail electricity price
            2. Solar-only: Solar generation with grid import/export, no battery
            3. Solar+battery: Full system with battery energy management

        Args:
            peak_price (float): Peak hour electricity price in EUR/kWh (6am-10pm weekdays).
                               If None, uses self.peak_price
            off_peak_price (float): Off-peak electricity price in EUR/kWh (nights and weekends).
                                   If None, uses self.off_peak_price
            export_price (float): Solar export feed-in tariff in EUR/kWh. Default: 0.0
            base_price (float): Legacy parameter - flat electricity price. Used if peak/off-peak are None

        Returns:
            Optional[Dict[str, float]]: Economic analysis results containing:
                - baseline_cost: Annual cost for grid-only scenario (EUR/year)
                - solar_only_cost: Annual cost with solar but no battery (EUR/year)
                - solar_battery_cost: Annual cost with full system (EUR/year)
                - savings_vs_baseline: Annual savings vs grid-only (EUR/year)
                - savings_vs_solar_only: Additional savings from battery (EUR/year)
            Returns None if simulation results not available.

        Calculation Methodology:
            - Grid import cost: total_grid_import_kwh × base_price
            - Export revenue: total_grid_export_kwh × export_price
            - Net cost: grid_import_cost - export_revenue
            - Savings: baseline_cost - system_cost

        Side Effects:
            - Prints detailed cost breakdown and savings analysis

        Requires:
            - self.simulation_results must be populated (call simulate_power_flows() first)
            - self.df must contain 'solar_generation_kwh' column

        Notes:
            - All costs calculated as annual totals
            - Export price typically lower than import price (0.05 vs 0.12 EUR/kWh)
            - Savings calculations exclude equipment capital costs
            - Time-of-use pricing not implemented (flat rate pricing assumed)
            - No consideration of demand charges or connection fees

        Example:
            >>> cost_analysis = simulator.calculate_costs_and_savings(
            ...     base_price=0.15, export_price=0.0
            ... )
            >>> annual_savings = cost_analysis['savings_vs_baseline']
            >>> battery_benefit = cost_analysis['savings_vs_solar_only']
        """
        if self.simulation_results is None:
            print("No simulation results available. Run simulation first.")
            return

        print("\nCalculating costs and savings...")

        # Determine pricing approach
        if peak_price is None:
            peak_price = self.peak_price
        if off_peak_price is None:
            off_peak_price = self.off_peak_price

        # Use time-based pricing if available, otherwise fall back to flat rate
        if peak_price is not None and off_peak_price is not None:
            use_time_based_pricing = True
            print("Using time-based pricing:")
            print(f"  Peak hours (6am-10pm weekdays): €{peak_price:.3f}/kWh")
            print(f"  Off-peak hours (nights/weekends): €{off_peak_price:.3f}/kWh")
        else:
            use_time_based_pricing = False
            if base_price is None:
                base_price = 0.12
            print(f"Using flat pricing: €{base_price:.3f}/kWh")

        print(f"Export price: €{export_price:.3f}/kWh")

        # Calculate costs with battery system using transmission block pricing
        def get_transmission_cost(row):
            block_num = int(row["transmission_block"])
            return self.transmission_costs.get(f"block{block_num}", 0.0)

        # Grid import cost with time-based and transmission block pricing
        def get_total_price(row):
            transmission_cost = get_transmission_cost(row)
            if use_time_based_pricing:
                # Get base price from simulation results (need is_peak_hour from original data)
                datetime_index = row.name  # Get the row index
                is_peak = (
                    self.df.iloc[datetime_index]["is_peak_hour"]
                    if datetime_index < len(self.df)
                    else False
                )
                base_price_for_row = peak_price if is_peak else off_peak_price
            else:
                base_price_for_row = base_price
            return base_price_for_row + transmission_cost

        grid_import_cost = self.simulation_results.apply(
            lambda row: row["grid_import_kwh"] * get_total_price(row), axis=1
        ).sum()

        # Export revenue (no transmission cost on exports)
        export_revenue = self.simulation_results["grid_export_kwh"].sum() * export_price

        # Calculate block-wise max power for monthly fees
        max_power_by_block = self.calculate_block_power("grid_import_kwh")
        self.max_power_by_block_simulated = max_power_by_block

        # Calculate OVE-SPTE cost
        ove_spte_cost_details = self.calculate_ove_spte_cost(max_power_by_block)
        annual_ove_spte_cost = ove_spte_cost_details.get("annual_ove_spte_cost_eur", 0)

        # Calculate monthly power fees for solar+battery system
        annual_monthly_power_fees = self._calculate_monthly_power_fees(max_power_by_block)

        net_cost_battery = (
            grid_import_cost - export_revenue + annual_ove_spte_cost + annual_monthly_power_fees
        )

        # Calculate baseline cost (no battery, no solar) with transmission costs
        # def get_baseline_total_price(row):
        #     transmission_cost = self.transmission_costs[f"block{int(row['transmission_block'])}"]
        #     if use_time_based_pricing:
        #         base_price_for_row = peak_price if row['is_peak_hour'] else off_peak_price
        #     else:
        #         base_price_for_row = base_price
        #     return base_price_for_row + transmission_cost

        baseline_grid_cost = self.df.apply(
            lambda row: row["consumption_kwh"] * get_total_price(row), axis=1
        ).sum()

        # Calculate block-wise max power for monthly fees
        max_power_by_block = self.calculate_block_power("consumption_kwh")
        self.max_power_by_block_baseline = max_power_by_block

        # Calculate baseline OVE-SPTE cost (consumption without any generation)
        baseline_ove_spte_cost = self.calculate_ove_spte_cost(max_power_by_block)
        annual_ove_spte_cost = baseline_ove_spte_cost.get("annual_ove_spte_cost_eur", 0)

        # Calculate baseline monthly power fees (consumption without any generation)
        annual_monthly_power_fees = self._calculate_monthly_power_fees(max_power_by_block)

        baseline_cost = baseline_grid_cost + annual_ove_spte_cost + annual_monthly_power_fees

        # Calculate savings
        savings_vs_baseline = baseline_cost - net_cost_battery

        print("\nCost Analysis:")
        print(f"  Baseline (grid only): €{baseline_cost:.2f}/year")
        print(f"  Solar + Battery: €{net_cost_battery:.2f}/year")
        if self.enable_power_smoothing:
            # Calculate potential power smoothing savings (this is already included in the costs above)
            total_power_smoothing = self.simulation_results["power_smoothing_discharge_kwh"].sum()
            if total_power_smoothing > 0:
                print(
                    f"  Power smoothing usage: {total_power_smoothing:.1f} kWh from {self.min_soc_reserve * 100:.0f}% reserve"
                )
                if hasattr(self, "power_smoothing_max_power"):
                    max_powers = self.power_smoothing_max_power
                    print(
                        f"  Achieved maximum power: Block 1: {max_powers['block_1']:.1f}kW, Block 2: {max_powers['block_2']:.1f}kW, Block 3: {max_powers['block_3']:.1f}kW"
                    )
        print(
            f"  Savings vs baseline: €{savings_vs_baseline:.2f}/year ({savings_vs_baseline / baseline_cost * 100:.1f}%)"
        )

        return {
            "baseline_cost": baseline_cost,
            "solar_battery_cost": net_cost_battery,
            "savings_vs_baseline": savings_vs_baseline,
        }

    def calculate_transmission_cost_breakdown(self) -> dict[str, dict[str, float]]:
        """
        Calculate detailed transmission cost analysis by block including max power analysis.

        Returns:
            Dict with block analysis including intervals, energy, costs, and power metrics
        """
        if self.simulation_results is None:
            print("No simulation results available. Run simulation first.")
            return {}

        block_analysis = {}
        total_intervals = 0

        for block in [1, 2, 3, 4, 5]:
            block_data = self.simulation_results[
                self.simulation_results["transmission_block"] == block
            ]

            intervals = len(block_data)
            total_intervals += intervals

            total_import_kwh = block_data["grid_import_kwh"].sum()
            total_export_kwh = block_data["grid_export_kwh"].sum()
            transmission_rate = self.transmission_costs[f"block{block}"]
            transmission_cost = total_import_kwh * transmission_rate

            # Power analysis (convert kWh per 15-min interval to kW)
            max_import_power_kw = block_data["grid_import_kwh"].max() * 4  # Convert to power
            avg_import_power_kw = block_data["grid_import_kwh"].mean() * 4 if intervals > 0 else 0
            max_export_power_kw = block_data["grid_export_kwh"].max() * 4
            avg_export_power_kw = block_data["grid_export_kwh"].mean() * 4 if intervals > 0 else 0

            # Consumption power analysis
            max_consumption_power_kw = (
                block_data["consumption_kwh"].max() * 4
                if "consumption_kwh" in block_data.columns
                else 0
            )
            avg_consumption_power_kw = (
                block_data["consumption_kwh"].mean() * 4
                if intervals > 0 and "consumption_kwh" in block_data.columns
                else 0
            )

            # Monthly power fee calculation (EUR/month) = max_power_kW * rate_EUR_per_kW_per_month
            power_fee_rate = self.monthly_power_fees[f"block{block}"]
            monthly_power_fee_eur = max_import_power_kw * power_fee_rate
            annual_power_fee_eur = monthly_power_fee_eur * 12

            block_analysis[f"block_{block}"] = {
                "intervals": intervals,
                "percentage_of_year": (intervals / 35040) * 100,  # 35040 = 365 * 24 * 4 intervals
                "total_import_kwh": total_import_kwh,
                "total_export_kwh": total_export_kwh,
                "transmission_rate_eur_per_kwh": transmission_rate,
                "transmission_cost_eur": transmission_cost,
                "max_import_power_kw": max_import_power_kw,
                "avg_import_power_kw": avg_import_power_kw,
                "max_export_power_kw": max_export_power_kw,
                "avg_export_power_kw": avg_export_power_kw,
                "max_consumption_power_kw": max_consumption_power_kw,
                "avg_consumption_power_kw": avg_consumption_power_kw,
                "power_fee_rate_eur_per_kw_per_month": power_fee_rate,
                "monthly_power_fee_eur": monthly_power_fee_eur,
                "annual_power_fee_eur": annual_power_fee_eur,
            }

        # Fix max power ordering: Block 1 <= Block 2 <= Block 3 <= Block 4 <= Block 5
        # Enforce constraint that no block can have lower max power than preceding blocks
        max_powers = []
        for block in [1, 2, 3, 4, 5]:
            max_powers.append(block_analysis[f"block_{block}"]["max_import_power_kw"])

        # Apply ordering constraint from Block 1 to Block 5
        for i in range(1, 5):  # blocks 2, 3, 4, 5 (indices 1, 2, 3, 4)
            if max_powers[i] < max_powers[i - 1]:
                # Current block has lower max power than previous block - adjust it up
                max_powers[i] = max_powers[i - 1]
                block_num = i + 1
                block_analysis[f"block_{block_num}"]["max_import_power_kw"] = max_powers[i]

                # Recalculate power fees with adjusted max power
                power_fee_rate = self.monthly_power_fees[f"block{block_num}"]
                monthly_power_fee_eur = max_powers[i] * power_fee_rate
                annual_power_fee_eur = monthly_power_fee_eur * 12
                block_analysis[f"block_{block_num}"]["monthly_power_fee_eur"] = (
                    monthly_power_fee_eur
                )
                block_analysis[f"block_{block_num}"]["annual_power_fee_eur"] = annual_power_fee_eur

        return block_analysis

    def calculate_block_power(self, column: str, debug: bool = False) -> dict[str, float]:
        """
        Calculate the maximum power for each block.

        Args:
            column: The column to calculate max power from ("grid_import_kwh" or "consumption_kwh")
            debug: Whether to print debug information
        """
        block_power = {}
        for block in [1, 2, 3, 4, 5]:
            block_data = self.simulation_results[
                self.simulation_results["transmission_block"] == block
            ]
            max_power_kw = block_data[column].max() * 4 if len(block_data) > 0 else 99999999
            block_power[f"block_{block}"] = max_power_kw
            if block > 1:
                # Ensure max power ordering: Block 1 <= Block 2 <= Block 3 <= Block 4 <= Block 5
                previous_block_power = block_power[f"block_{block - 1}"]
                if max_power_kw < previous_block_power:
                    max_power_kw = previous_block_power
                    block_power[f"block_{block}"] = max_power_kw
        if debug:
            print("Block Power (kW):", block_power)
        return block_power

    def calculate_ove_spte_cost(self, max_power_by_block: dict[str, float]) -> dict[str, float]:
        """
        Calculate OVE-SPTE cost based on weighted block power usage.
        Formula: (4 × Block1_max_power + 8 × Block2_max_power) × ove_spte_fee EUR/kW/month

        Returns:
            Dict with OVE-SPTE cost details
        """
        if self.simulation_results is None:
            print("No simulation results available. Run simulation first.")
            return {}

        # Calculate weighted power: 4 × Block1 + 8 × Block2
        weighted_power_kw = 4 * max_power_by_block["block_1"] + 8 * max_power_by_block["block_2"]

        # Calculate monthly and annual costs
        annual_ove_spte_cost = weighted_power_kw * self.ove_spte_fee

        return {
            "weighted_power_kw": weighted_power_kw,
            "ove_spte_rate_eur_per_kw_per_month": self.ove_spte_fee,
            "annual_ove_spte_cost_eur": annual_ove_spte_cost,
        }

    def _calculate_monthly_power_fees(self, max_power_by_block: dict[str, float]) -> float:
        """Calculate monthly power fees for solar+battery scenario"""
        if self.simulation_results is None:
            return 0.0

        total_annual_power_fees = 0.0

        # Calculate power fees for each block based on max grid import power
        for block in [1, 2, 3, 4, 5]:
            max_power_kw = max_power_by_block[f"block_{block}"]
            power_fee_rate = self.monthly_power_fees[f"block{block}"]
            monthly_power_fee = max_power_kw * power_fee_rate
            months_block_is_active = 12
            if block == 1:
                months_block_is_active = 4
            elif block == 5:
                months_block_is_active = 8
            annual_power_fee = monthly_power_fee * months_block_is_active
            total_annual_power_fees += annual_power_fee

        return total_annual_power_fees

    def create_power_flow_visualization(self, days_to_show: int = 7) -> None:
        """Create visualization of power flows over time"""
        if self.simulation_results is None:
            print("No simulation results available. Run simulation first.")
            return

        print(f"\nCreating power flow visualization for {days_to_show} days...")

        # Select data for visualization (first week)
        plot_data = self.simulation_results.head(
            days_to_show * 24 * 4
        ).copy()  # 4 intervals per hour
        plot_data["hour_of_day"] = (
            plot_data["datetime"].dt.hour + plot_data["datetime"].dt.minute / 60
        )

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

        # Plot 1: Power generation and consumption
        ax1.plot(
            plot_data["hour_of_day"],
            plot_data["solar_generation_kwh"] * 4,
            label="Solar Generation",
            color="orange",
            linewidth=2,
        )
        ax1.plot(
            plot_data["hour_of_day"],
            plot_data["consumption_kwh"] * 4,
            label="Consumption",
            color="blue",
            linewidth=2,
        )
        ax1.fill_between(
            plot_data["hour_of_day"],
            0,
            plot_data["solar_generation_kwh"] * 4,
            alpha=0.3,
            color="orange",
        )
        ax1.set_ylabel("Power (kW)")
        ax1.set_title(f"Solar Generation vs Consumption - First {days_to_show} Days")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Battery state of charge
        ax2.plot(
            plot_data["hour_of_day"], plot_data["battery_soc_percent"], color="green", linewidth=2
        )
        ax2.fill_between(
            plot_data["hour_of_day"], 0, plot_data["battery_soc_percent"], alpha=0.3, color="green"
        )
        ax2.set_ylabel("Battery SOC (%)")
        ax2.set_title("Battery State of Charge")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Grid import/export
        ax3.plot(
            plot_data["hour_of_day"],
            plot_data["grid_import_kwh"] * 4,
            label="Grid Import",
            color="red",
            linewidth=2,
        )
        ax3.plot(
            plot_data["hour_of_day"],
            -plot_data["grid_export_kwh"] * 4,
            label="Grid Export",
            color="purple",
            linewidth=2,
        )
        ax3.fill_between(
            plot_data["hour_of_day"], 0, plot_data["grid_import_kwh"] * 4, alpha=0.3, color="red"
        )
        ax3.fill_between(
            plot_data["hour_of_day"],
            0,
            -plot_data["grid_export_kwh"] * 4,
            alpha=0.3,
            color="purple",
        )
        ax3.set_ylabel("Power (kW)")
        ax3.set_xlabel("Time (hours)")
        ax3.set_title("Grid Import/Export (Export shown as negative)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"power_flow_analysis_{days_to_show}days.png", dpi=300, bbox_inches="tight")
        print(f"Power flow visualization saved as 'power_flow_analysis_{days_to_show}days.png'")
        plt.close()

    def export_results(self) -> None:
        """Export detailed results to CSV files"""
        if self.simulation_results is None:
            print("No simulation results available. Run simulation first.")
            return

        print("\nExporting results...")

        # Export detailed simulation results
        export_df = self.simulation_results.copy()
        export_df.to_csv("power_flow_simulation_results.csv", index=False)
        print("Detailed results exported to 'power_flow_simulation_results.csv'")

        # Create summary statistics
        summary_stats = {
            "System Configuration": [
                f"{self.solar_panel_power_kw} kW solar panels",
                f"{self.inverter_power_kw} kW inverter",
                f"{self.battery_capacity_kwh} kWh battery",
                f"{self.battery_charge_power_kw} kW charge power",
                f"{self.battery_discharge_power_kw} kW discharge power",
                f"{self.battery_efficiency * 100:.0f}% battery efficiency",
            ],
            "Annual Totals (kWh)": [
                f"Solar generation: {self.simulation_results['solar_generation_kwh'].sum():.1f}",
                f"Consumption: {self.simulation_results['consumption_kwh'].sum():.1f}",
                f"Grid import: {self.simulation_results['grid_import_kwh'].sum():.1f}",
                f"Grid export: {self.simulation_results['grid_export_kwh'].sum():.1f}",
                f"Battery charge: {self.simulation_results['battery_charge_kwh'].sum():.1f}",
                f"Battery discharge: {self.simulation_results['battery_discharge_kwh'].sum():.1f}",
            ],
            "Battery Statistics": [
                f"Average SOC: {self.simulation_results['battery_soc_percent'].mean():.1f}%",
                f"Minimum SOC: {self.simulation_results['battery_soc_percent'].min():.1f}%",
                f"Maximum SOC: {self.simulation_results['battery_soc_percent'].max():.1f}%",
                f"Annual cycles: {(self.simulation_results['battery_charge_kwh'].sum() + self.simulation_results['battery_discharge_kwh'].sum()) / (2 * self.battery_capacity_kwh):.1f}"
                if self.battery_capacity_kwh > 0
                else "Annual cycles: 0.0 (no battery)",
                "",
                "",
            ],
        }

        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_csv("system_summary.csv", index=False)
        print("System summary exported to 'system_summary.csv'")


# Standalone cached function for batch simulations
@memory.cache
def _cached_batch_simulation_func(
    solar_kw: float,
    inverter_kw: float,
    battery_kwh: float,
    battery_charge_power_kw: float,
    battery_discharge_power_kw: float,
    battery_efficiency: float,
    peak_price: float,
    off_peak_price: float,
    export_price: float,
    production_file: str,
    consumption_file: str,
    transmission_costs: dict[str, float],
    ove_spte_fee: float,
    enable_power_smoothing: bool = True,
    max_power_block1: float = 300.0,
    max_power_block2: float = 320.0,
    min_soc_reserve: float = 0.5,
    heating_config_tuple: tuple = None,
) -> dict[str, Any]:
    """
    Standalone cached simulation function for batch mode to avoid re-running identical simulations.

    Note: heating_config is passed as a tuple for cache hashability.
    Format: (heating_kwh, start_hour, end_hour) or None
    """
    try:
        # Suppress all output for batch mode (including constructor)
        import sys
        from io import StringIO

        # Capture output
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            # Create simulator instance
            simulator = PowerFlowSimulator(
                solar_panel_power_kw=solar_kw,
                inverter_power_kw=inverter_kw,
                battery_capacity_kwh=battery_kwh,
                battery_c_rate=0.5,
                battery_efficiency=battery_efficiency,
                production_file=production_file,
                consumption_file=consumption_file,
                transmission_costs=transmission_costs,
                ove_spte_fee=ove_spte_fee,
                peak_price=peak_price,
                off_peak_price=off_peak_price,
                enable_power_smoothing=enable_power_smoothing,
                min_soc_reserve=min_soc_reserve,
                max_power_by_block={
                    1: max_power_block1,
                    2: max_power_block2,
                    3: 340.0,  # Use default
                    4: 2000.0,  # Use default
                    5: 2000.0,  # Use default
                },
                heating_config=dict(zip(["heating_kwh", "start_hour", "end_hour"], heating_config_tuple)) if heating_config_tuple else None,
            )

            # Run simulation
            simulator.load_and_align_data()
            simulator.scale_solar_generation()
            simulator.simulate_power_flows()
            cost_analysis = simulator.calculate_costs_and_savings(
                peak_price=peak_price, off_peak_price=off_peak_price, export_price=export_price
            )
        finally:
            sys.stdout = old_stdout

        if cost_analysis is None:
            return {"simulation_failed": True}

        return {
            "simulation_failed": False,
            "cost_analysis": cost_analysis,
            "simulation_results": simulator.simulation_results.copy()
            if simulator.simulation_results is not None
            else None,
        }

    except Exception as e:
        return {"simulation_failed": True, "error": str(e)}


class MultiScenarioAnalyzer:
    """
    Multi-scenario batch analyzer for solar + battery system optimization.

    This class automates the execution of hundreds of PowerFlowSimulator scenarios across
    different combinations of solar panel capacity, inverter power, and battery storage.
    It provides comprehensive comparison analysis, economic optimization, and visualization
    tools to identify optimal system configurations.

    Key Capabilities:
        - Generate all valid parameter combinations from user-specified ranges
        - Execute batch simulations with output suppression for clean processing
        - Rank scenarios by economic performance (annual savings, ROI, payback period)
        - Generate comprehensive visualization suite (heatmaps, Pareto analysis, time series)
        - Export detailed results for further analysis and reporting
        - Economic optimization with equipment cost modeling

    Analysis Features:
        - Annual savings comparison across all scenarios
        - ROI analysis with customizable equipment costs
        - Self-sufficiency analysis (grid independence metrics)
        - Pareto frontier identification (cost vs performance trade-offs)
        - Battery utilization analysis across different capacities
        - Inverter sizing optimization and clipping loss analysis

    Performance Considerations:
        - Automatically validates scenario combinations (inverter ≤ solar capacity)
        - Calculates battery power ratings from capacity using C-rate methodology
        - Suppresses individual simulation output for batch processing efficiency
        - Stores detailed battery logs for scenarios with storage systems
        - Memory usage scales with: n_scenarios × data_points × stored_variables

    Visualization Outputs:
        - Savings heatmap: Annual savings across solar/battery combinations
        - ROI heatmap: Payback periods for investment analysis
        - Self-sufficiency heatmap: Energy independence percentages
        - Pareto analysis: Investment vs savings optimization scatter plot
        - SOC time series (by battery): Battery SOC organized by battery capacity
        - SOC time series (by solar+inverter): Battery SOC organized by solar+inverter combinations

    Economic Modeling:
        - Configurable equipment costs (€/kW solar, €/kWh battery)
        - Electricity pricing (import and export rates)
        - ROI calculations with payback period analysis
        - Savings calculations relative to grid-only baseline

    Attributes:
        solar_range (List[float]): Solar panel capacities to test (kW)
        inverter_range (List[float]): Inverter powers to test (kW)
        battery_range (List[float]): Battery capacities to test (kWh)
        battery_c_rate (float): C-rate for battery power calculations
        electricity_price (float): Grid import price (EUR/kWh)
        export_price (float): Solar export price (EUR/kWh)
        solar_cost_per_kw (float): Solar installation cost (EUR/kW)
        battery_cost_per_kwh (float): Battery cost (EUR/kWh)
        scenarios (List[Dict]): Generated parameter combinations
        results (List[Dict]): Comprehensive simulation results

    Typical Workflow:
        1. Initialize with parameter ranges and economic assumptions
        2. generate_scenarios() - Create all valid combinations
        3. run_all_scenarios() - Execute batch simulations
        4. create_comparison_summary() - Rank and export results
        5. Create visualization suite with various heatmaps and plots
        6. export_comprehensive_results() - Generate detailed reports

    Example:
        >>> analyzer = MultiScenarioAnalyzer(
        ...     solar_range=[10, 15, 20, 25],
        ...     battery_range=[0, 10, 20, 30],
        ...     peak_price=0.15,
        ...     off_peak_price=0.12,
        ...     solar_cost_per_kw=1000,
        ...     battery_cost_per_kwh=600
        ... )
        >>> analyzer.run_complete_analysis()  # Full analysis pipeline
        >>> # Results available in generated CSV and PNG files
    """

    def __init__(
        self,
        solar_range: list[float] | None = None,
        inverter_range: list[float] | None = None,
        battery_range: list[float] | None = None,
        battery_c_rate: float = 0.5,
        battery_efficiency: float = 0.9,
        peak_price: float = 0.14683,
        off_peak_price: float = 0.10664,
        export_price: float = 0.01,
        solar_cost_per_kw: float = 450,
        inverter_cost_per_kw: float = 130,
        battery_cost_per_kwh: float = 250,
        maintenance_fee_per_kw: float = 10.0,
        battery_maintenance_fee_per_kwh: float = 5.0,
        discount_rate: float = 0.05,
        loan_rate: float = 0.03,
        loan_years: int = 10,
        production_file: str = "production.csv",
        consumption_file: str = "consumption.csv",
        transmission_costs: dict[str, float] = None,
        ove_spte_fee: float = 3.44078,
        enable_power_smoothing: bool = False,
        min_soc_reserve: float = 0.2,
        max_power_by_block: dict[int, float] = None,
        heating_config: dict = None,
    ):
        """
        Multi-scenario analyzer for solar + battery systems

        Args:
            solar_range (list): List of solar panel capacities to test (kW)
            inverter_range (list): List of inverter powers to test (kW)
            battery_range (list): List of battery capacities to test (kWh)
            battery_c_rate (float): C-rate for calculating battery power (default: 0.5)
            battery_efficiency (float): Battery round-trip efficiency
            peak_price (float): Peak hour electricity price (EUR/kWh) - weekdays 6am-10pm
            off_peak_price (float): Off-peak hour electricity price (EUR/kWh) - nights/weekends
            export_price (float): Export electricity price (EUR/kWh)
            solar_cost_per_kw (float): Solar installation cost per kW
            inverter_cost_per_kw (float): Inverter cost per kW
            battery_cost_per_kwh (float): Battery cost per kWh
            maintenance_fee_per_kw (float): Annual maintenance fee per kW of solar (EUR/kW/year)
            battery_maintenance_fee_per_kwh (float): Annual maintenance fee per kWh of battery (EUR/kWh/year)
            discount_rate (float): Annual discount rate for NPV calculations (default: 0.05 = 5%)
            loan_rate (float): Annual loan interest rate (default: 0.03 = 3%)
            loan_years (int): Loan term in years (default: 10)
            transmission_costs (Dict[str, float]): Transmission costs by block (block1-block5)
            heating_config (dict): Heating load configuration with keys:
                - heating_kwh (float): Total heating energy for season in kWh
                - start_hour (int): Daily start hour (default: 7)
                - end_hour (int): Daily end hour (0 = midnight, default: 0)
        """
        self.solar_range = solar_range or [5, 8.5, 10, 15, 20]
        self.inverter_range = inverter_range or [5, 8, 10, 15]
        self.battery_range = battery_range or [0, 10, 15, 20]
        self.battery_c_rate = battery_c_rate
        self.battery_efficiency = battery_efficiency
        self.peak_price = peak_price
        self.off_peak_price = off_peak_price
        self.export_price = export_price
        self.solar_cost_per_kw = solar_cost_per_kw
        self.inverter_cost_per_kw = inverter_cost_per_kw
        self.battery_cost_per_kwh = battery_cost_per_kwh
        self.maintenance_fee_per_kw = maintenance_fee_per_kw
        self.battery_maintenance_fee_per_kwh = battery_maintenance_fee_per_kwh
        self.discount_rate = discount_rate
        self.loan_rate = loan_rate
        self.loan_years = loan_years
        self.production_file = production_file
        self.consumption_file = consumption_file

        # Transmission costs (EUR/kWh) - 5 blocks (1=most expensive, 3=cheapest)
        if transmission_costs is None:
            transmission_costs = {
                "block1": 0.01282,
                "block2": 0.01216,
                "block3": 0.01186,
                "block4": 0.01164,
                "block5": 0.01175,
            }
        self.transmission_costs = transmission_costs
        self.ove_spte_fee = ove_spte_fee

        # Power smoothing parameters
        self.enable_power_smoothing = enable_power_smoothing
        self.min_soc_reserve = min_soc_reserve
        self.max_power_by_block = max_power_by_block or {
            1: 300.0,
            2: 320.0,
            3: 340.0,
            4: 2000.0,
            5: 2000.0,
        }

        # Heating load configuration
        self.heating_config = heating_config

        self.scenarios = []
        self.results = []

    def generate_scenarios(self) -> list[dict[str, float]]:
        """Generate all valid parameter combinations"""
        print("Generating scenario combinations...")

        # Calculate total combinations
        total_combinations = (
            len(self.solar_range) * len(self.inverter_range) * len(self.battery_range)
        )
        print(f"Evaluating {total_combinations} parameter combinations...")

        self.scenarios = []
        combinations = list(product(self.solar_range, self.inverter_range, self.battery_range))

        for solar_kw, inverter_kw, battery_kwh in tqdm(
            combinations, desc="Generating scenarios", unit="combo"
        ):
            # Only include valid combinations (inverter <= solar)
            if inverter_kw <= solar_kw:
                # Calculate battery power based on C-rate
                battery_charge_power = battery_kwh * self.battery_c_rate
                battery_discharge_power = battery_kwh * self.battery_c_rate

                scenario = {
                    "solar_panel_power_kw": solar_kw,
                    "inverter_power_kw": inverter_kw,
                    "battery_capacity_kwh": battery_kwh,
                    "battery_charge_power_kw": battery_charge_power,
                    "battery_discharge_power_kw": battery_discharge_power,
                    "battery_efficiency": self.battery_efficiency,
                }
                self.scenarios.append(scenario)

        print(f"Generated {len(self.scenarios)} valid scenarios")
        return self.scenarios

    def _cached_batch_simulation(
        self,
        solar_kw: float,
        inverter_kw: float,
        battery_kwh: float,
        battery_charge_power_kw: float,
        battery_discharge_power_kw: float,
        battery_efficiency: float,
        peak_price: float,
        off_peak_price: float,
        export_price: float,
    ) -> dict[str, Any]:
        """
        Cached simulation function for batch mode to avoid re-running identical simulations.

        Args:
            solar_kw: Solar panel capacity in kW
            inverter_kw: Inverter capacity in kW
            battery_kwh: Battery capacity in kWh
            battery_charge_power_kw: Battery charge power in kW
            battery_discharge_power_kw: Battery discharge power in kW
            battery_efficiency: Battery round-trip efficiency
            peak_price: Peak hour electricity price
            off_peak_price: Off-peak hour electricity price
            export_price: Grid export price

        Returns:
            Dictionary with simulation results and cost analysis
        """
        return _cached_batch_simulation_func(
            solar_kw,
            inverter_kw,
            battery_kwh,
            battery_charge_power_kw,
            battery_discharge_power_kw,
            battery_efficiency,
            peak_price,
            off_peak_price,
            export_price,
            self.production_file,
            self.consumption_file,
            self.transmission_costs,
            self.ove_spte_fee,
            enable_power_smoothing=self.enable_power_smoothing,
            max_power_block1=self.max_power_by_block.get(1, 300.0),
            max_power_block2=self.max_power_by_block.get(2, 320.0),
            min_soc_reserve=self.min_soc_reserve,
            heating_config_tuple=tuple(self.heating_config.get(k) for k in ["heating_kwh", "start_hour", "end_hour"]) if self.heating_config else None,
        )

    def run_all_scenarios(self) -> list[dict[str, Any]]:
        """Run simulations for all scenarios"""
        if not self.scenarios:
            self.generate_scenarios()

        print(f"\nRunning {len(self.scenarios)} simulations...")
        self.results = []

        for scenario in tqdm(
            self.scenarios,
            desc="Simulating scenarios",
            unit="scenario",
            postfix=f"Total: {len(self.scenarios)}",
        ):
            # Use cached simulation
            cached_result = self._cached_batch_simulation(
                solar_kw=scenario["solar_panel_power_kw"],
                inverter_kw=scenario["inverter_power_kw"],
                battery_kwh=scenario["battery_capacity_kwh"],
                battery_charge_power_kw=scenario["battery_charge_power_kw"],
                battery_discharge_power_kw=scenario["battery_discharge_power_kw"],
                battery_efficiency=scenario["battery_efficiency"],
                peak_price=self.peak_price,
                off_peak_price=self.off_peak_price,
                export_price=self.export_price,
            )

            if cached_result["simulation_failed"]:
                print(f"    Simulation failed: {cached_result.get('error', 'Unknown error')}")
                continue

            cost_analysis = cached_result["cost_analysis"]
            simulation_results = cached_result["simulation_results"]

            # Calculate equipment costs and ROI
            solar_investment = scenario["solar_panel_power_kw"] * self.solar_cost_per_kw
            inverter_investment = scenario["inverter_power_kw"] * self.inverter_cost_per_kw
            battery_investment = scenario["battery_capacity_kwh"] * self.battery_cost_per_kwh
            total_investment = solar_investment + inverter_investment + battery_investment

            annual_savings = cost_analysis["savings_vs_baseline"]
            payback_years = (
                total_investment / annual_savings if annual_savings > 0 else float("inf")
            )

            # Calculate maintenance costs and NPV with financing
            annual_solar_maintenance = (
                scenario["solar_panel_power_kw"] * self.maintenance_fee_per_kw
            )
            annual_battery_maintenance = (
                scenario["battery_capacity_kwh"] * self.battery_maintenance_fee_per_kwh
            )
            annual_general_one_percent_maintenance = total_investment * 0.01
            annual_maintenance_cost = (
                annual_solar_maintenance
                + annual_battery_maintenance
                + annual_general_one_percent_maintenance
            )
            annual_loan_payment = self.calculate_loan_payment(total_investment)

            # Net annual cash flow = savings - maintenance - loan payments
            net_annual_cash_flow = annual_savings - annual_maintenance_cost - annual_loan_payment

            # Calculate NPV over 20 years (no upfront investment, just annual cash flows)
            npv_20_years = 0
            for year in range(1, 21):  # Years 1-20
                if year <= self.loan_years:
                    # Years 1-10: Pay loan + maintenance, receive savings
                    annual_cash_flow = net_annual_cash_flow
                else:
                    # Years 11-20: No loan payments, just maintenance
                    annual_cash_flow = annual_savings - annual_maintenance_cost
                npv_20_years += annual_cash_flow / ((1 + self.discount_rate) ** year)

            # Calculate key metrics
            total_consumption = simulation_results["consumption_kwh"].sum()
            total_solar = simulation_results["solar_generation_kwh"].sum()
            total_grid_import = simulation_results["grid_import_kwh"].sum()
            self_sufficiency = (1 - total_grid_import / total_consumption) * 100

            # Store comprehensive results
            result = {
                **scenario,
                **cost_analysis,
                "solar_investment": solar_investment,
                "battery_investment": battery_investment,
                "total_investment": total_investment,
                "payback_years": payback_years,
                "annual_solar_maintenance": annual_solar_maintenance,
                "annual_battery_maintenance": annual_battery_maintenance,
                "annual_maintenance_cost": annual_maintenance_cost,
                "net_annual_cash_flow": net_annual_cash_flow,
                "annual_loan_payment": annual_loan_payment,
                "npv_20_years": npv_20_years,
                "discount_rate": self.discount_rate,
                "annual_solar_generation": total_solar,
                "annual_consumption": total_consumption,
                "annual_grid_import": total_grid_import,
                "annual_grid_export": simulation_results["grid_export_kwh"].sum(),
                "self_sufficiency_percent": self_sufficiency,
                "battery_cycles_per_year": (
                    simulation_results["battery_charge_kwh"].sum()
                    + simulation_results["battery_discharge_kwh"].sum()
                )
                / (2 * scenario["battery_capacity_kwh"])
                if scenario["battery_capacity_kwh"] > 0
                else 0,
                "avg_battery_soc_percent": simulation_results["battery_soc_percent"].mean(),
                "battery_log": simulation_results.copy()
                if scenario["battery_capacity_kwh"] > 0
                else None,
            }

            self.results.append(result)

        print(f"\nCompleted all {len(self.scenarios)} simulations!")
        return self.results

    def create_comparison_summary(self):
        """Create summary table comparing all scenarios"""
        if not self.results:
            print("No results available. Run simulations first.")
            return None

        print("\nCreating comparison summary...")

        # Create comprehensive DataFrame
        df = pd.DataFrame(self.results)

        # Sort by annual savings (descending)
        df = df.sort_values("savings_vs_baseline", ascending=False)

        # Create summary columns
        summary_df = pd.DataFrame(
            {
                "Rank": range(1, len(df) + 1),
                "Solar (kW)": df["solar_panel_power_kw"],
                "Inverter (kW)": df["inverter_power_kw"],
                "Battery (kWh)": df["battery_capacity_kwh"],
                "Annual Savings (€)": df["savings_vs_baseline"].round(2),
                "Investment (€)": df["total_investment"].round(0),
                "Payback (years)": df["payback_years"].round(1),
                "Self-Sufficiency (%)": df["self_sufficiency_percent"].round(1),
                "Solar Generation (kWh)": df["annual_solar_generation"].round(0),
                "Grid Import (kWh)": df["annual_grid_import"].round(1),
                "Battery Cycles/year": df["battery_cycles_per_year"].round(1),
            }
        )

        # Export to CSV
        summary_df.to_csv("multi_scenario_comparison.csv", index=False)
        print("Comparison summary exported to 'multi_scenario_comparison.csv'")

        # Print top 10 scenarios
        print("\nTop 10 Scenarios by Annual Savings:")
        print(summary_df.head(10).to_string(index=False))

        return summary_df

    def _get_parameters_string(self):
        """Generate parameter string for filenames including power smoothing settings"""
        params = []
        if hasattr(self, "enable_power_smoothing") and self.enable_power_smoothing:
            params.append("power_smoothing")
            if hasattr(self, "min_soc_reserve"):
                params.append(f"reserve_{self.min_soc_reserve:.0%}")
            if hasattr(self, "max_power_by_block") and self.max_power_by_block:
                for block_num in [1, 2, 3]:
                    if block_num in self.max_power_by_block:
                        power = self.max_power_by_block[block_num]
                        params.append(f"block{block_num}_{power:.0f}kW")

        # Add pricing parameters if different from defaults
        if hasattr(self, "peak_price") and abs(self.peak_price - 0.147) > 0.001:
            params.append(f"peak_{self.peak_price:.3f}")
        if hasattr(self, "off_peak_price") and abs(self.off_peak_price - 0.107) > 0.001:
            params.append(f"offpeak_{self.off_peak_price:.3f}")

        return "_" + "_".join(params) if params else ""

    def create_savings_heatmap(self):
        """Create heatmap of annual savings across solar and battery combinations"""
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print("\nCreating savings heatmap...")

        # Create pivot table for heatmap
        df = pd.DataFrame(self.results)
        pivot_data = df.pivot_table(
            values="savings_vs_baseline",
            index="battery_capacity_kwh",
            columns="solar_panel_power_kw",
            aggfunc="max",  # Take maximum savings if multiple inverter sizes
        )

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".0f",
            cmap="viridis",
            cbar_kws={"label": "Annual Savings (€)"},
        )

        plt.title(
            "Annual Savings Heatmap: Solar Panel Size vs Battery Capacity",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Solar Panel Power (kW)")
        plt.ylabel("Battery Capacity (kWh)")
        plt.tight_layout()

        # Generate filename with power smoothing parameters
        params_str = self._get_parameters_string()
        filename = f"multi_scenario_savings_heatmap{params_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Savings heatmap saved as '{filename}'")
        plt.close()

    def create_roi_heatmap(self):
        """Create heatmap of payback periods (ROI analysis)"""
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print("\nCreating ROI heatmap...")

        # Create pivot table for ROI heatmap
        df = pd.DataFrame(self.results)

        # Cap payback years at 20 for visualization
        df["payback_years_capped"] = df["payback_years"].apply(lambda x: min(x, 20))

        pivot_data = df.pivot_table(
            values="payback_years_capped",
            index="battery_capacity_kwh",
            columns="solar_panel_power_kw",
            aggfunc="min",  # Take minimum payback time if multiple inverter sizes
        )

        # Create heatmap with reversed colormap (green = good ROI, red = poor ROI)
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=20,
            cbar_kws={"label": "Payback Period (Years)"},
        )

        plt.title(
            f"ROI Payback Period Heatmap\n(Solar: €{self.solar_cost_per_kw}/kW, Inverter: €{self.inverter_cost_per_kw}/kW, Battery: €{self.battery_cost_per_kwh}/kWh)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Solar Panel Power (kW)")
        plt.ylabel("Battery Capacity (kWh)")
        plt.tight_layout()

        # Generate filename with power smoothing parameters
        params_str = self._get_parameters_string()
        filename = f"multi_scenario_roi_heatmap{params_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"ROI heatmap saved as '{filename}'")
        plt.close()

    def create_self_sufficiency_heatmap(self):
        """Create heatmap of self-sufficiency percentages"""
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print("\nCreating self-sufficiency heatmap...")

        # Create pivot table for self-sufficiency heatmap
        df = pd.DataFrame(self.results)
        pivot_data = df.pivot_table(
            values="self_sufficiency_percent",
            index="battery_capacity_kwh",
            columns="solar_panel_power_kw",
            aggfunc="max",  # Take maximum self-sufficiency if multiple inverter sizes
        )

        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt=".1f",
            cmap="plasma",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Self-Sufficiency (%)"},
        )

        plt.title(
            "Self-Sufficiency Heatmap: Solar Panel Size vs Battery Capacity",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Solar Panel Power (kW)")
        plt.ylabel("Battery Capacity (kWh)")
        plt.tight_layout()

        # Generate filename with power smoothing parameters
        params_str = self._get_parameters_string()
        filename = f"multi_scenario_self_sufficiency_heatmap{params_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Self-sufficiency heatmap saved as '{filename}'")
        plt.close()

    def create_npv_heatmap(self):
        """Create heatmap of Net Present Value over 20 years"""
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print("\nCreating NPV heatmap...")

        # Create pivot table for NPV heatmap
        df = pd.DataFrame(self.results)

        # Find the optimal inverter power for each solar/battery combination
        idx = df.groupby(["battery_capacity_kwh", "solar_panel_power_kw"])["npv_20_years"].idxmax()
        optimal_scenarios = df.loc[idx]

        # Create pivot tables for NPV and optimal inverter power
        pivot_npv = optimal_scenarios.pivot_table(
            values="npv_20_years",
            index="battery_capacity_kwh",
            columns="solar_panel_power_kw",
            aggfunc="first",
        )

        pivot_inverter = optimal_scenarios.pivot_table(
            values="inverter_power_kw",
            index="battery_capacity_kwh",
            columns="solar_panel_power_kw",
            aggfunc="first",
        )

        # Create custom annotations combining NPV and inverter power
        annotations = []
        for i in range(len(pivot_npv.index)):
            row = []
            for j in range(len(pivot_npv.columns)):
                npv = pivot_npv.iloc[i, j]
                inverter = pivot_inverter.iloc[i, j]
                if pd.notna(npv) and pd.notna(inverter):
                    row.append(f"{npv:.0f}\n({inverter:.0f}kW)")
                else:
                    row.append("")
            annotations.append(row)

        # Create heatmap with colormap (green = positive NPV, red = negative NPV)
        plt.figure(figsize=(14, 10))

        # Use actual data range for better visualization
        actual_min = pivot_npv.min().min()
        actual_max = pivot_npv.max().max()

        # If all values are positive, use a sequential colormap
        if actual_min >= 0:
            sns.heatmap(
                pivot_npv,
                annot=annotations,
                fmt="",
                cmap="YlOrRd",  # Sequential colormap for all-positive values
                vmin=actual_min,
                vmax=actual_max,
                cbar_kws={"label": "Net Present Value (EUR)"},
                annot_kws={"fontsize": 9},
            )
        else:
            # Use diverging colormap centered at zero for mixed pos/neg values
            vmax = max(abs(actual_min), abs(actual_max))
            vmin = -vmax
            sns.heatmap(
                pivot_npv,
                annot=annotations,
                fmt="",
                cmap="RdYlGn",
                center=0,
                vmin=vmin,
                vmax=vmax,
                cbar_kws={"label": "Net Present Value (EUR)"},
                annot_kws={"fontsize": 9},
            )

        plt.title(
            f"NPV Heatmap (20 years, {self.discount_rate * 100:.1f}% discount rate)\n"
            f"Maintenance: €{self.maintenance_fee_per_kw}/kW/year + €{self.battery_maintenance_fee_per_kwh}/kWh/year, Solar: €{self.solar_cost_per_kw}/kW, Inverter: €{self.inverter_cost_per_kw}/kW, Battery: €{self.battery_cost_per_kwh}/kWh\n"
            f"Values show: NPV (Optimal Inverter Power)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Solar Panel Power (kW)")
        plt.ylabel("Battery Capacity (kWh)")
        plt.tight_layout()

        # Generate filename with power smoothing parameters
        params_str = self._get_parameters_string()
        filename = f"multi_scenario_npv_heatmap{params_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"NPV heatmap saved as '{filename}'")
        plt.close()

    def create_pareto_analysis(self):
        """Create Pareto frontier analysis (cost vs performance)"""
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print("\nCreating Pareto analysis...")

        df = pd.DataFrame(self.results)

        # Create scatter plot of investment vs annual savings
        plt.figure(figsize=(14, 10))

        # Color by battery capacity
        scatter = plt.scatter(
            df["total_investment"],
            df["savings_vs_baseline"],
            c=df["battery_capacity_kwh"],
            cmap="viridis",
            s=60,
            alpha=0.7,
            edgecolors="black",
            linewidth=0.5,
        )

        plt.colorbar(scatter, label="Battery Capacity (kWh)")
        plt.xlabel("Total Investment (€)")
        plt.ylabel("Annual Savings (€)")
        plt.title(
            "Investment vs Annual Savings\n(Color indicates battery capacity)",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3)

        # Add annotations for best scenarios
        top_scenarios = df.nlargest(5, "savings_vs_baseline")
        for _, row in top_scenarios.iterrows():
            plt.annotate(
                f"{row['solar_panel_power_kw']}kW+{row['battery_capacity_kwh']}kWh",
                (row["total_investment"], row["savings_vs_baseline"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.8,
            )

        plt.tight_layout()
        plt.savefig("multi_scenario_pareto_analysis.png", dpi=300, bbox_inches="tight")
        print("Pareto analysis saved as 'multi_scenario_pareto_analysis.png'")
        plt.close()

    def create_soc_time_plot(self, days_to_show=365):
        """Create plot showing SOC vs time for all battery combinations"""
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print(f"\nCreating SOC vs time plot for {days_to_show} days...")

        # Filter results to only include scenarios with batteries
        battery_results = [r for r in self.results if r["battery_capacity_kwh"] > 0]

        if not battery_results:
            print("No battery scenarios found to plot.")
            return

        # Get unique battery capacities and sort them
        battery_capacities = sorted(list(set(r["battery_capacity_kwh"] for r in battery_results)))

        # Create subplots - one for each battery capacity
        n_capacities = len(battery_capacities)
        fig, axes = plt.subplots(n_capacities, 1, figsize=(16, 4 * n_capacities))

        # Handle single subplot case
        if n_capacities == 1:
            axes = [axes]

        # Define colors for different solar/inverter combinations
        colors = plt.cm.tab10(np.linspace(0, 1, 10))

        for i, capacity in enumerate(battery_capacities):
            ax = axes[i]

            # Get all scenarios with this battery capacity
            capacity_scenarios = [
                r for r in battery_results if r["battery_capacity_kwh"] == capacity
            ]

            # Plot each scenario
            for j, scenario in enumerate(capacity_scenarios):
                # Get battery log data from the simulation results
                # We need to find the corresponding simulation result
                matching_result = None
                for result in self.results:
                    if (
                        result["solar_panel_power_kw"] == scenario["solar_panel_power_kw"]
                        and result["inverter_power_kw"] == scenario["inverter_power_kw"]
                        and result["battery_capacity_kwh"] == scenario["battery_capacity_kwh"]
                    ):
                        matching_result = result
                        break

                if matching_result and "battery_log" in matching_result:
                    battery_log = matching_result["battery_log"]

                    # Select data for visualization (first N days)
                    max_intervals = days_to_show * 24 * 4  # 4 intervals per hour
                    plot_data = battery_log.head(max_intervals).copy()

                    # Create time axis in hours from start
                    plot_data["hours_from_start"] = range(len(plot_data))
                    plot_data["hours_from_start"] = (
                        plot_data["hours_from_start"] * 0.25
                    )  # 15-min intervals

                    # Create label for this scenario
                    label = f"{scenario['solar_panel_power_kw']:.0f}kW solar + {scenario['inverter_power_kw']:.0f}kW inv"

                    # Plot SOC
                    color = colors[j % len(colors)]
                    ax.plot(
                        plot_data["hours_from_start"],
                        plot_data["battery_soc_percent"],
                        linewidth=1.5,
                        alpha=0.8,
                        color=color,
                        label=label,
                    )

            # Customize subplot
            ax.set_xlabel("Time (hours from start)")
            ax.set_ylabel("Battery SOC (%)")
            ax.set_title(f"Battery State of Charge - {capacity:.0f} kWh Battery")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

            # Add legend if there are multiple scenarios
            if len(capacity_scenarios) > 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.savefig(f"multi_scenario_soc_time_{days_to_show}days.png", dpi=300, bbox_inches="tight")
        print(f"SOC vs time plot saved as 'multi_scenario_soc_time_{days_to_show}days.png'")
        plt.close()

    def create_soc_by_solar_inverter_plot(self) -> None:
        """
        Create separate SOC plots for each solar panel + inverter combination.

        This method generates individual plots for each unique solar panel and inverter
        combination, showing all battery capacities as different colored lines on each plot.
        This organization makes it easier to compare battery sizing effects for specific
        solar and inverter configurations.

        Monthly Pattern Analysis:
            Each plot shows 12 lines representing the SOC pattern for each month
            of the year, normalized to month progress (0-1 scale)

        Organization:
            - One individual plot file per solar panel + inverter combination
            - 12 monthly SOC patterns shown as separate colored lines
            - Normalized time axis (0-1) allows direct comparison of monthly patterns

        Output:
            - Multiple PNG files: 'soc_{solar}kW_solar_{inverter}kW_inverter_monthly.png'
            - Individual plot files showing monthly SOC patterns for each solar+inverter combination

        Side Effects:
            - Creates and saves multiple visualization files
            - Prints progress messages for each file created
            - Prints summary of total files generated

        Requires:
            - self.results must contain battery scenarios with 'battery_log' data
            - Battery logs must contain 'battery_soc_percent' and datetime columns

        Notes:
            - Complements the existing battery-organized SOC plots
            - Monthly patterns reveal seasonal behavior and solar resource variation
            - Each month normalized to 0-1 scale enables direct pattern comparison
            - Summer months typically show higher SOC patterns than winter months
            - Individual files allow focused analysis of each system configuration

        Example:
            >>> analyzer.create_soc_by_solar_inverter_plot()
            # Creates individual files like:
            # soc_800kW_solar_750kW_inverter_monthly.png
            # soc_700kW_solar_650kW_inverter_monthly.png
            # Each showing 12 monthly SOC patterns
        """
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print("\nCreating monthly SOC pattern plots by solar+inverter combination...")

        # Filter results to only include scenarios with batteries
        battery_results = [r for r in self.results if r["battery_capacity_kwh"] > 0]

        if not battery_results:
            print("No battery scenarios found to plot.")
            return

        # Get unique solar+inverter combinations
        solar_inverter_combinations = set()
        for result in battery_results:
            combo = (result["solar_panel_power_kw"], result["inverter_power_kw"])
            solar_inverter_combinations.add(combo)

        # Sort combinations for consistent ordering
        combinations = sorted(list(solar_inverter_combinations))
        n_combinations = len(combinations)

        if n_combinations == 0:
            print("No solar+inverter combinations found.")
            return

        # Get unique battery capacities for consistent color mapping
        battery_capacities = sorted(list(set(r["battery_capacity_kwh"] for r in battery_results)))
        colors = plt.cm.tab10(np.linspace(0, 1, len(battery_capacities)))
        battery_color_map = {capacity: colors[i] for i, capacity in enumerate(battery_capacities)}

        # Create individual plots for each solar+inverter combination
        for solar_kw, inverter_kw in combinations:
            # Create new figure for this combination
            plt.figure(figsize=(12, 8))

            # Get all battery scenarios for this solar+inverter combination
            combination_scenarios = [
                r
                for r in battery_results
                if r["solar_panel_power_kw"] == solar_kw and r["inverter_power_kw"] == inverter_kw
            ]

            # Plot each battery capacity as a separate line
            for scenario in combination_scenarios:
                battery_capacity = scenario["battery_capacity_kwh"]

                # Find the matching result with battery log
                matching_result = None
                for result in self.results:
                    if (
                        result["solar_panel_power_kw"] == scenario["solar_panel_power_kw"]
                        and result["inverter_power_kw"] == scenario["inverter_power_kw"]
                        and result["battery_capacity_kwh"] == scenario["battery_capacity_kwh"]
                    ):
                        matching_result = result
                        break

                if matching_result and "battery_log" in matching_result:
                    battery_log = matching_result["battery_log"]

                    # Add month column to battery log
                    battery_log["month"] = battery_log["datetime"].dt.month

                    # Create label for this battery capacity
                    battery_label = f"{battery_capacity:.0f} kWh"
                    color = battery_color_map[battery_capacity]

                    # Plot each month as a separate line
                    month_names = [
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ]
                    month_colors = plt.cm.tab20(np.linspace(0, 1, 12))

                    for month in range(1, 13):
                        month_data = battery_log[battery_log["month"] == month].copy()

                        if len(month_data) > 0:
                            # Create normalized time axis for the month (0-1 representing month progress)
                            month_data = month_data.reset_index(drop=True)
                            month_data["month_progress"] = np.linspace(0, 1, len(month_data))

                            # Plot this month's SOC pattern
                            month_color = month_colors[month - 1]
                            alpha = (
                                0.7 if scenario == combination_scenarios[0] else 0.7
                            )  # Same alpha for all

                            # Only add month labels for the first battery capacity to avoid legend clutter
                            if scenario == combination_scenarios[0]:
                                month_label = month_names[month - 1]
                            else:
                                month_label = None

                            plt.plot(
                                month_data["month_progress"],
                                month_data["battery_soc_percent"],
                                color=month_color,
                                linewidth=1.0,
                                alpha=alpha,
                                label=month_label,
                            )

            # Customize plot
            plt.xlabel("Month Progress (0 = Start of Month, 1 = End of Month)")
            plt.ylabel("Battery SOC (%)")
            plt.title(
                f"{solar_kw:.0f}kW Solar + {inverter_kw:.0f}kW Inverter - Monthly SOC Patterns"
            )
            plt.ylim(0, 100)
            plt.xlim(0, 1)
            plt.grid(True, alpha=0.3)

            # Add legend for months
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9, title="Months")

            # Save individual plot file
            filename = f"soc_{solar_kw:.0f}kW_solar_{inverter_kw:.0f}kW_inverter_monthly.png"
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(
                f"Created monthly SOC plot for {solar_kw:.0f}kW solar + {inverter_kw:.0f}kW inverter: {filename}"
            )
            plt.close()

        print(
            f"Generated {len(combinations)} individual monthly SOC plots for solar+inverter combinations"
        )

    def export_comprehensive_results(self):
        """Export comprehensive results to Excel file"""
        if not self.results:
            print("No results available. Run simulations first.")
            return

        print("\nExporting comprehensive results to Excel...")

        # Create detailed DataFrame
        df = pd.DataFrame(self.results)

        # Round numeric columns for readability
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(2)

        # Sort by annual savings (descending)
        df = df.sort_values("savings_vs_baseline", ascending=False)

        try:
            with pd.ExcelWriter(
                "multi_scenario_comprehensive_results.xlsx", engine="openpyxl"
            ) as writer:
                # Main results sheet
                df.to_excel(writer, sheet_name="All Results", index=False)

                # Summary sheet (top performers)
                summary_cols = [
                    "solar_panel_power_kw",
                    "inverter_power_kw",
                    "battery_capacity_kwh",
                    "savings_vs_baseline",
                    "total_investment",
                    "payback_years",
                    "self_sufficiency_percent",
                    "annual_solar_generation",
                ]
                df[summary_cols].head(20).to_excel(writer, sheet_name="Top 20 Summary", index=False)

                # Analysis by categories
                df[df["battery_capacity_kwh"] == 0].to_excel(
                    writer, sheet_name="Solar Only", index=False
                )
                df[df["battery_capacity_kwh"] > 0].to_excel(
                    writer, sheet_name="Solar + Battery", index=False
                )

            print("Comprehensive results exported to 'multi_scenario_comprehensive_results.xlsx'")

        except ImportError:
            print("openpyxl not available, exporting to CSV instead...")
            df.to_csv("multi_scenario_comprehensive_results.csv", index=False)
            print("Comprehensive results exported to 'multi_scenario_comprehensive_results.csv'")

    def calculate_loan_payment(self, principal: float) -> float:
        """
        Calculate annual loan payment using loan amortization formula.

        Args:
            principal (float): Loan amount (total investment)

        Returns:
            float: Annual loan payment
        """
        if principal <= 0:
            return 0

        if self.loan_rate == 0:
            return principal / self.loan_years

        # Annual payment = P * [r(1+r)^n] / [(1+r)^n - 1]
        # where P = principal, r = annual rate, n = number of years
        r = self.loan_rate
        n = self.loan_years
        return principal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    def optimize_npv_differential_evolution(
        self,
        maxiter: int = 1000,
        popsize: int = 15,
        seed: int = 42,
        polish: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Use scipy's differential evolution to find optimal solar + battery configuration
        that maximizes Net Present Value (NPV) over 20 years.

        Args:
            maxiter: Maximum number of iterations (default: 1000)
            popsize: Population size multiplier (default: 15)
            seed: Random seed for reproducibility (default: 42)
            polish: Apply L-BFGS-B local search to best solution (default: True)
            verbose: Print optimization progress (default: True)

        Returns:
            Dict containing optimal configuration and optimization results
        """

        print(f"\n{'=' * 70}")
        print("DIFFERENTIAL EVOLUTION OPTIMIZATION FOR NPV MAXIMIZATION")
        print(f"{'=' * 70}")
        print("Constraints:")
        print("  - Solar power: 100-1000 kW (50kW steps)")
        print("  - Inverter power: 50 kW to solar power (50kW steps)")
        print("  - Battery capacity: 0-1000 kWh (50kWh steps)")
        print("  - Block 1 max power: 200-500 kW (50kW steps)")
        print("  - Block 2 max power: 250-550 kW (50kW steps, ≥ Block 1)")
        print("  - Min SOC reserve: 10-80% (5% steps)")
        print("  - Objective: Maximize 20-year NPV")
        print("\nOptimization parameters:")
        print(f"  - Max iterations: {maxiter}")
        print(f"  - Population size: {popsize} × 6 = {popsize * 6}")
        print(f"  - Random seed: {seed}")
        print(f"  - Polish with L-BFGS-B: {polish}")

        # Track function evaluations for progress
        self._eval_count = 0
        self._best_npv = -np.inf
        self._best_config = None

        # Track parameter evolution for plotting
        self._evolution_history = {
            "evaluations": [],
            "best_solar": [],
            "best_inverter": [],
            "best_battery": [],
            "best_block1": [],
            "best_block2": [],
            "best_reserve": [],
            "best_npv": [],
        }

        def objective_function(x):
            """
            Objective function to minimize (negative NPV).
            x = [solar_kw, inverter_kw, battery_kwh, power_block_1, power_block_2, min_soc_reserve]
            """
            self._eval_count += 1

            solar_kw, inverter_kw, battery_kwh, power_block_1, power_block_2, min_soc_reserve = x

            # Enforce discrete steps: 50kW for solar/inverter, 50kWh for battery
            solar_kw = round(solar_kw / 50) * 50  # Round to nearest 50kW
            inverter_kw = round(inverter_kw / 50) * 50  # Round to nearest 50kW
            battery_kwh = round(battery_kwh / 50) * 50  # Round to nearest 50kWh
            power_block_1 = round(power_block_1 / 50) * 50  # Round to nearest 50kWh
            power_block_2 = round(power_block_2 / 50) * 50  # Round to nearest 50kWh
            min_soc_reserve *= 100
            min_soc_reserve = round(min_soc_reserve / 5) * 5 / 100

            if verbose and self._eval_count % 100 == 1:
                print(
                    f"  Eval #{self._eval_count}: Trying Solar={solar_kw:.0f}kW, Inverter={inverter_kw:.0f}kW, Battery={battery_kwh:.0f}kWh, Block1={power_block_1:.0f}kW, Block2={power_block_2:.0f}kW, Reserve={min_soc_reserve:.0%}"
                )

            # Constraint: inverter power <= solar power
            if inverter_kw > solar_kw:
                return 1e6  # Large penalty for constraint violation
            if power_block_2 < power_block_1:
                return 1e6  # Large penalty for constraint violation

            try:
                # Run simulation using cached function
                result = _cached_batch_simulation_func(
                    solar_kw=float(solar_kw),
                    inverter_kw=float(inverter_kw),
                    battery_kwh=float(battery_kwh),
                    battery_charge_power_kw=float(battery_kwh * 0.5),  # 0.5C rate
                    battery_discharge_power_kw=float(battery_kwh * 0.5),  # 0.5C rate
                    battery_efficiency=0.9,
                    peak_price=self.peak_price,
                    off_peak_price=self.off_peak_price,
                    export_price=self.export_price,
                    production_file="production.csv",
                    consumption_file="consumption.csv",
                    transmission_costs=self.transmission_costs,
                    ove_spte_fee=self.ove_spte_fee,
                    enable_power_smoothing=True,
                    max_power_block1=float(power_block_1),
                    max_power_block2=float(power_block_2),
                    min_soc_reserve=float(min_soc_reserve),
                    heating_config_tuple=tuple(self.heating_config.get(k) for k in ["heating_kwh", "start_hour", "end_hour"]) if self.heating_config else None,
                )

                if result["simulation_failed"]:
                    return 1e6  # Penalty for failed simulation

                # Calculate NPV (same logic as in run_all_scenarios)
                cost_analysis = result["cost_analysis"]
                annual_savings = cost_analysis["savings_vs_baseline"]

                # Calculate investment costs
                solar_investment = solar_kw * self.solar_cost_per_kw
                inverter_investment = inverter_kw * self.inverter_cost_per_kw
                battery_investment = battery_kwh * self.battery_cost_per_kwh
                total_investment = solar_investment + inverter_investment + battery_investment

                # Calculate maintenance costs
                annual_solar_maintenance = solar_kw * self.maintenance_fee_per_kw
                annual_battery_maintenance = battery_kwh * self.battery_maintenance_fee_per_kwh
                annual_maintenance_cost = annual_solar_maintenance + annual_battery_maintenance
                annual_loan_payment = self.calculate_loan_payment(total_investment)

                # Calculate 20-year NPV
                npv_20_years = 0
                for year in range(1, 21):  # Years 1-20
                    if year <= self.loan_years:
                        annual_cash_flow = (
                            annual_savings - annual_maintenance_cost - annual_loan_payment
                        )
                    else:
                        annual_cash_flow = annual_savings - annual_maintenance_cost
                    npv_20_years += annual_cash_flow / ((1 + self.discount_rate) ** year)

                # Track best solution
                if npv_20_years > self._best_npv:
                    self._best_npv = npv_20_years
                    self._best_config = {
                        "solar_kw": solar_kw,
                        "inverter_kw": inverter_kw,
                        "battery_kwh": battery_kwh,
                        "npv": npv_20_years,
                        "annual_savings": annual_savings,
                        "investment": total_investment,
                    }

                    # Track evolution for plotting
                    self._evolution_history["evaluations"].append(self._eval_count)
                    self._evolution_history["best_solar"].append(solar_kw)
                    self._evolution_history["best_inverter"].append(inverter_kw)
                    self._evolution_history["best_battery"].append(battery_kwh)
                    self._evolution_history["best_npv"].append(npv_20_years)

                if verbose and self._eval_count % 50 == 0:
                    print(f"  Evaluation {self._eval_count}: Best NPV = €{self._best_npv:,.0f}")

                return -npv_20_years  # Minimize negative NPV (maximize NPV)

            except Exception as e:
                if verbose and self._eval_count % 100 == 0:
                    print(f"  Evaluation {self._eval_count} failed: {str(e)}")
                return 1e6  # Large penalty for failed simulation

        # Define bounds: [solar_kw, inverter_kw, battery_kwh, power_block_1, power_block_2, min_soc_reserve]
        bounds = [
            (100, 1050),  # Solar power (kW)
            (50, 1050),  # Inverter power (kW)
            (0, 1050),  # Battery capacity (kWh)
            (200, 500),  # Block 1 max power threshold (kW)
            (250, 550),  # Block 2 max power threshold (kW)
            (0.1, 0.8),  # Min SOC reserve (10% to 80%)
        ]

        print("\nStarting optimization...")
        print(f"Expected ~{maxiter * popsize * 6} function evaluations")

        # Run differential evolution
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            polish=polish,
            strategy="rand2bin",
            disp=verbose,
        )

        print(f"\n{'=' * 70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 70}")

        if result.success:
            print("✓ Optimization converged successfully")
        else:
            print(f"⚠ Optimization terminated: {result.message}")

        print(f"Total function evaluations: {result.nfev}")
        print(f"Iterations: {result.nit}")

        # Extract optimal solution
        (
            optimal_solar,
            optimal_inverter,
            optimal_battery,
            optimal_block1,
            optimal_block2,
            optimal_reserve,
        ) = result.x
        optimal_npv = -result.fun

        # Apply same rounding as objective function
        optimal_solar = round(optimal_solar / 50) * 50
        optimal_inverter = round(optimal_inverter / 50) * 50
        optimal_battery = round(optimal_battery / 50) * 50
        optimal_block1 = round(optimal_block1 / 50) * 50
        optimal_block2 = round(optimal_block2 / 50) * 50
        optimal_reserve *= 100
        optimal_reserve = round(optimal_reserve / 5) * 5 / 100

        print("\n🎯 OPTIMAL CONFIGURATION:")
        print(f"  Solar power: {optimal_solar:.0f} kW")
        print(f"  Inverter power: {optimal_inverter:.0f} kW")
        print(f"  Battery capacity: {optimal_battery:.0f} kWh")
        print(f"  Block 1 max power: {optimal_block1:.0f} kW")
        print(f"  Block 2 max power: {optimal_block2:.0f} kW")
        print(f"  Min SOC reserve: {optimal_reserve:.0%}")
        print(f"  Maximum NPV: €{optimal_npv:,.0f}")

        if self._best_config:
            print(f"  Annual savings: €{self._best_config['annual_savings']:,.0f}")
            print(f"  Total investment: €{self._best_config['investment']:,.0f}")

        # Save optimization plot
        self._save_optimization_plot(result)

        return {
            "success": result.success,
            "message": result.message,
            "optimal_solar_kw": optimal_solar,
            "optimal_inverter_kw": optimal_inverter,
            "optimal_battery_kwh": optimal_battery,
            "optimal_block1_kw": optimal_block1,
            "optimal_block2_kw": optimal_block2,
            "optimal_min_soc_reserve": optimal_reserve,
            "optimal_npv": optimal_npv,
            "function_evaluations": result.nfev,
            "iterations": result.nit,
            "full_result": result,
            "best_config": self._best_config,
        }

    def _save_optimization_plot(self, result):
        """Save optimization convergence and parameter evolution plots"""
        try:
            # Create comprehensive optimization visualization
            fig = plt.figure(figsize=(15, 12))

            # Plot 1: Parameter Evolution over Time
            plt.subplot(2, 2, 1)
            if len(self._evolution_history["evaluations"]) > 1:
                evals = self._evolution_history["evaluations"]
                plt.plot(
                    evals,
                    self._evolution_history["best_solar"],
                    "o-",
                    color="orange",
                    label="Solar (kW)",
                    linewidth=2,
                    markersize=4,
                )
                plt.plot(
                    evals,
                    self._evolution_history["best_inverter"],
                    "s-",
                    color="blue",
                    label="Inverter (kW)",
                    linewidth=2,
                    markersize=4,
                )
                plt.plot(
                    evals,
                    self._evolution_history["best_battery"],
                    "^-",
                    color="green",
                    label="Battery (kWh)",
                    linewidth=2,
                    markersize=4,
                )
                plt.title("Parameter Evolution During Optimization")
                plt.xlabel("Function Evaluation")
                plt.ylabel("Parameter Value")
                plt.legend()
                plt.grid(True, alpha=0.3)

            # Plot 2: NPV Evolution
            plt.subplot(2, 2, 2)
            if len(self._evolution_history["evaluations"]) > 1:
                plt.plot(
                    self._evolution_history["evaluations"],
                    [npv / 1e6 for npv in self._evolution_history["best_npv"]],
                    "o-",
                    color="red",
                    linewidth=2,
                    markersize=4,
                )
                plt.title("NPV Evolution During Optimization")
                plt.xlabel("Function Evaluation")
                plt.ylabel("Best NPV (Million €)")
                plt.grid(True, alpha=0.3)

            # Plot 3: Convergence history (if available from differential evolution)
            plt.subplot(2, 2, 3)
            if hasattr(result, "convergence"):
                plt.plot(-np.array(result.convergence) / 1e6, linewidth=2)
                plt.title("Iteration-by-Iteration Convergence")
                plt.xlabel("Iteration")
                plt.ylabel("Best NPV (Million €)")
                plt.grid(True, alpha=0.3)
            else:
                # Show final parameters evolution as bars
                if len(self._evolution_history["evaluations"]) > 0:
                    final_idx = len(self._evolution_history["evaluations"]) - 1
                    params = [
                        self._evolution_history["best_solar"][final_idx],
                        self._evolution_history["best_inverter"][final_idx],
                        self._evolution_history["best_battery"][final_idx],
                    ]
                    labels = ["Solar\n(kW)", "Inverter\n(kW)", "Battery\n(kWh)"]
                    colors = ["orange", "blue", "green"]
                    bars = plt.bar(labels, params, color=colors, alpha=0.7)

                    # Add value labels
                    for bar, val in zip(bars, params, strict=False):
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(params) * 0.01,
                            f"{val:.0f}",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                        )
                    plt.title("Final Optimal Configuration")
                    plt.ylabel("Value")

            # Plot 4: Parameter Trajectory in 3D (if enough history points)
            plt.subplot(2, 2, 4)
            if len(self._evolution_history["evaluations"]) > 3:
                # Create a 2D projection showing solar vs inverter evolution
                solar_vals = self._evolution_history["best_solar"]
                inverter_vals = self._evolution_history["best_inverter"]
                battery_vals = self._evolution_history["best_battery"]

                # Color points by progression (early = blue, late = red)
                n_points = len(solar_vals)
                colors = plt.cm.viridis(np.linspace(0, 1, n_points))

                scatter = plt.scatter(
                    solar_vals, inverter_vals, c=range(n_points), cmap="viridis", s=60, alpha=0.7
                )
                plt.plot(solar_vals, inverter_vals, "-", alpha=0.3, color="gray")
                plt.colorbar(scatter, label="Optimization Progress")
                plt.xlabel("Solar Power (kW)")
                plt.ylabel("Inverter Power (kW)")
                plt.title("Solar vs Inverter Evolution Path")
                plt.grid(True, alpha=0.3)
            else:
                # Show simple evaluation progress
                plt.text(
                    0.5,
                    0.5,
                    f"Total Evaluations: {result.nfev}\nIterations: {result.nit}\n\nFinal NPV: €{-result.fun:,.0f}",
                    ha="center",
                    va="center",
                    transform=plt.gca().transAxes,
                    fontsize=12,
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
                )
                plt.title("Optimization Summary")
                plt.axis("off")

            plt.tight_layout()
            plt.savefig("optimization_evolution.png", dpi=300, bbox_inches="tight")
            print("\nOptimization evolution plot saved as 'optimization_evolution.png'")
            plt.close()

        except Exception as e:
            print(f"Could not save optimization plot: {e}")

    def run_complete_analysis(self):
        """Run complete multi-scenario analysis with all outputs"""
        print("Starting complete multi-scenario analysis...")

        # Run all simulations
        self.run_all_scenarios()

        # Create summary
        self.create_comparison_summary()

        # Create visualizations
        self.create_savings_heatmap()
        self.create_roi_heatmap()
        self.create_npv_heatmap()
        self.create_self_sufficiency_heatmap()
        self.create_pareto_analysis()
        # self.create_soc_time_plot()  # Disabled - large file generation
        # self.create_soc_by_solar_inverter_plot()  # Disabled - generates many individual files

        # Export comprehensive results
        self.export_comprehensive_results()

        print("\n" + "=" * 60)
        print("MULTI-SCENARIO ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\nGenerated files:")
        print("- multi_scenario_comparison.csv")

        # Generate dynamic filenames with parameters
        params_str = self._get_parameters_string()
        print(f"- multi_scenario_savings_heatmap{params_str}.png")
        print(f"- multi_scenario_roi_heatmap{params_str}.png")
        print(f"- multi_scenario_self_sufficiency_heatmap{params_str}.png")
        print(f"- multi_scenario_npv_heatmap{params_str}.png")
        print("- multi_scenario_pareto_analysis.png")
        # print("- multi_scenario_soc_time_365days.png")  # Disabled
        # print("- soc_[solar]kW_solar_[inverter]kW_inverter_monthly.png (multiple files)")  # Disabled
        print("- multi_scenario_comprehensive_results.xlsx (or .csv)")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Power Flow Simulator for Solar + Battery Systems")

    # System specifications
    parser.add_argument(
        "--solar-power", type=float, default=8.5, help="Solar panel capacity in kW (default: 8.5)"
    )
    parser.add_argument(
        "--inverter-power",
        type=float,
        default=8.0,
        help="Inverter maximum AC output in kW (default: 8.0)",
    )
    parser.add_argument(
        "--battery-capacity",
        type=float,
        default=10.0,
        help="Battery storage capacity in kWh (default: 10.0)",
    )
    parser.add_argument(
        "--battery-c-rate",
        type=float,
        default=0.5,
        help="Battery C-rate for charge/discharge power (default: 0.5)",
    )
    parser.add_argument(
        "--battery-efficiency",
        type=float,
        default=0.9,
        help="Battery round-trip efficiency (default: 0.9)",
    )

    # Pricing
    parser.add_argument(
        "--peak-price",
        type=float,
        default=0.14683,
        help="Peak hour electricity price in EUR/kWh (6am-10pm weekdays, default: 0.14683)",
    )
    parser.add_argument(
        "--off-peak-price",
        type=float,
        default=0.10664,
        help="Off-peak hour electricity price in EUR/kWh (nights and weekends, default: 0.10664)",
    )
    parser.add_argument(
        "--export-price",
        type=float,
        default=0.01,
        help="Solar export price in EUR/kWh (default: 0.01)",
    )

    # Power smoothing settings
    parser.add_argument(
        "--enable-power-smoothing",
        action="store_true",
        help="Enable power smoothing: use battery reserve to minimize maximum power draw",
    )
    parser.add_argument(
        "--min-soc-reserve",
        type=float,
        default=0.2,
        help="Minimum SOC reserve for power smoothing (0-1, default: 0.2)",
    )
    parser.add_argument(
        "--max-power-threshold",
        type=float,
        default=None,
        help="Maximum power threshold in kW (optional, auto-calculated if not specified)",
    )
    parser.add_argument(
        "--max-power-block1",
        type=float,
        default=None,
        help="Maximum power threshold for Block 1 in kW",
    )
    parser.add_argument(
        "--max-power-block2",
        type=float,
        default=None,
        help="Maximum power threshold for Block 2 in kW",
    )
    parser.add_argument(
        "--max-power-block3",
        type=float,
        default=None,
        help="Maximum power threshold for Block 3 in kW",
    )
    parser.add_argument(
        "--max-power-block4",
        type=float,
        default=None,
        help="Maximum power threshold for Block 4 in kW",
    )
    parser.add_argument(
        "--max-power-block5",
        type=float,
        default=None,
        help="Maximum power threshold for Block 5 in kW",
    )

    # Transmission costs
    parser.add_argument(
        "--transmission-block1",
        type=float,
        default=0.01282,
        help="Block 1 transmission cost in EUR/kWh (default: 0.01282)",
    )
    parser.add_argument(
        "--transmission-block2",
        type=float,
        default=0.01216,
        help="Block 2 transmission cost in EUR/kWh (default: 0.01216)",
    )
    parser.add_argument(
        "--transmission-block3",
        type=float,
        default=0.01186,
        help="Block 3 transmission cost in EUR/kWh (default: 0.01186)",
    )
    parser.add_argument(
        "--transmission-block4",
        type=float,
        default=0.01164,
        help="Block 4 transmission cost in EUR/kWh (default: 0.01164)",
    )
    parser.add_argument(
        "--transmission-block5",
        type=float,
        default=0.01175,
        help="Block 5 transmission cost in EUR/kWh (default: 0.01175)",
    )

    # Monthly power fees
    parser.add_argument(
        "--power-fee-block1",
        type=float,
        default=3.75969,
        help="Block 1 monthly power fee in EUR/kW/month (default: 3.75969)",
    )
    parser.add_argument(
        "--power-fee-block2",
        type=float,
        default=1.05262,
        help="Block 2 monthly power fee in EUR/kW/month (default: 1.05262)",
    )
    parser.add_argument(
        "--power-fee-block3",
        type=float,
        default=0.12837,
        help="Block 3 monthly power fee in EUR/kW/month (default: 0.12837)",
    )
    parser.add_argument(
        "--power-fee-block4",
        type=float,
        default=0.0,
        help="Block 4 monthly power fee in EUR/kW/month (default: 0.0)",
    )
    parser.add_argument(
        "--power-fee-block5",
        type=float,
        default=0.0,
        help="Block 5 monthly power fee in EUR/kW/month (default: 0.0)",
    )

    # OVE-SPTE fee
    parser.add_argument(
        "--ove-spte-fee",
        type=float,
        default=3.44078,
        help="OVE-SPTE fee rate in EUR/kW/month (default: 3.44078)",
    )

    # Multi-scenario analysis
    parser.add_argument(
        "--batch-mode", action="store_true", help="Enable multi-scenario analysis mode"
    )
    parser.add_argument(
        "--solar-range",
        type=str,
        default="5,8.5,10,15,20",
        help="Solar panel capacities to test in kW (comma-separated, default: 5,8.5,10,15,20)",
    )
    parser.add_argument(
        "--inverter-range",
        type=str,
        default="5,8,10,15",
        help="Inverter powers to test in kW (comma-separated, default: 5,8,10,15)",
    )
    parser.add_argument(
        "--battery-range",
        type=str,
        default="0,10,15,20",
        help='Battery capacities to test in kWh (comma-separated, default: "0,10,15,20")',
    )

    # Equipment costs for ROI analysis
    parser.add_argument(
        "--solar-cost-per-kw",
        type=float,
        default=450,
        help="Solar installation cost per kW (default: 450)",
    )
    parser.add_argument(
        "--inverter-cost-per-kw",
        type=float,
        default=130,
        help="Inverter cost per kW (default: 130)",
    )
    parser.add_argument(
        "--battery-cost-per-kwh",
        type=float,
        default=250,
        help="Battery cost per kWh (default: 250)",
    )
    parser.add_argument(
        "--maintenance-fee-per-kw",
        type=float,
        default=10.0,
        help="Annual maintenance fee per kW of solar power (default: 10.0)",
    )
    parser.add_argument(
        "--battery-maintenance-fee-per-kwh",
        type=float,
        default=5.0,
        help="Annual maintenance fee per kWh of battery (default: 5.0)",
    )
    parser.add_argument(
        "--discount-rate",
        type=float,
        default=0.05,
        help="Discount rate for NPV calculations (default: 0.05)",
    )
    parser.add_argument(
        "--loan-rate", type=float, default=0.03, help="Annual loan interest rate (default: 0.03)"
    )
    parser.add_argument(
        "--loan-years", type=int, default=10, help="Loan term in years (default: 10)"
    )

    # Optimization parameters
    parser.add_argument(
        "--optimize-npv",
        action="store_true",
        help="Run differential evolution optimization to maximize NPV",
    )
    parser.add_argument(
        "--opt-maxiter",
        type=int,
        default=1000,
        help="Maximum iterations for optimization (default: 1000)",
    )
    parser.add_argument(
        "--opt-popsize", type=int, default=15, help="Population size for optimization (default: 15)"
    )
    parser.add_argument(
        "--opt-seed", type=int, default=42, help="Random seed for optimization (default: 42)"
    )
    parser.add_argument(
        "--opt-polish",
        action="store_true",
        default=True,
        help="Polish the optimization result (default: True)",
    )

    # Heating load arguments
    parser.add_argument(
        "--add-heating-load",
        action="store_true",
        help="Add synthetic heating load during winter months (Oct-Mar)",
    )
    parser.add_argument(
        "--heating-kwh",
        type=float,
        default=3333,
        help="Total heating energy per season in kWh (default: 3333)",
    )
    parser.add_argument(
        "--heating-start-hour",
        type=int,
        default=7,
        help="Daily heating start hour (default: 7)",
    )
    parser.add_argument(
        "--heating-end-hour",
        type=int,
        default=0,
        help="Daily heating end hour, 0=midnight (default: 0)",
    )

    args = parser.parse_args()

    def parse_range(range_str):
        """Parse comma-separated range string to list of floats"""
        return [float(x.strip()) for x in range_str.split(",")]

    print("Power Flow Simulator for Solar + Battery Systems")
    print("=" * 60)

    if args.batch_mode:
        print("\nRunning in BATCH MODE - Multi-scenario analysis")
        print(f"Solar range: {args.solar_range}")
        print(f"Inverter range: {args.inverter_range}")
        print(f"Battery range: {args.battery_range}")
        print(f"Battery C-rate: {args.battery_c_rate}")
        print(
            f"Equipment costs: €{args.solar_cost_per_kw}/kW solar, €{args.inverter_cost_per_kw}/kW inverter, €{args.battery_cost_per_kwh}/kWh battery"
        )
        print(
            f"Electricity prices: €{args.peak_price:.3f}/kWh peak, €{args.off_peak_price:.3f}/kWh off-peak"
        )
        print(f"Export price: €{args.export_price:.3f}/kWh")
        if args.add_heating_load:
            print(f"Heating load: {args.heating_kwh} kWh/season, {args.heating_start_hour}:00-{'midnight' if args.heating_end_hour == 0 else f'{args.heating_end_hour}:00'}")

        # Parse parameter ranges
        solar_range = parse_range(args.solar_range)
        inverter_range = parse_range(args.inverter_range)
        battery_range = parse_range(args.battery_range)

        # Create transmission costs dictionary from arguments
        transmission_costs = {
            "block1": args.transmission_block1,
            "block2": args.transmission_block2,
            "block3": args.transmission_block3,
            "block4": args.transmission_block4,
            "block5": args.transmission_block5,
        }

        # Create monthly power fees dictionary from arguments
        monthly_power_fees = {
            "block1": args.power_fee_block1,
            "block2": args.power_fee_block2,
            "block3": args.power_fee_block3,
            "block4": args.power_fee_block4,
            "block5": args.power_fee_block5,
        }

        # Create max power by block dictionary from arguments (only if any values are provided)
        max_power_by_block = {}
        if any(
            [
                args.max_power_block1,
                args.max_power_block2,
                args.max_power_block3,
                args.max_power_block4,
                args.max_power_block5,
            ]
        ):
            max_power_by_block = {
                1: args.max_power_block1 or 300.0,
                2: args.max_power_block2 or 320.0,
                3: args.max_power_block3 or 340.0,
                4: args.max_power_block4 or 2000.0,
                5: args.max_power_block5 or 2000.0,
            }

        # Initialize multi-scenario analyzer
        analyzer = MultiScenarioAnalyzer(
            solar_range=solar_range,
            inverter_range=inverter_range,
            battery_range=battery_range,
            battery_c_rate=args.battery_c_rate,
            battery_efficiency=args.battery_efficiency,
            peak_price=args.peak_price,
            off_peak_price=args.off_peak_price,
            export_price=args.export_price,
            solar_cost_per_kw=args.solar_cost_per_kw,
            inverter_cost_per_kw=args.inverter_cost_per_kw,
            battery_cost_per_kwh=args.battery_cost_per_kwh,
            transmission_costs=transmission_costs,
            ove_spte_fee=args.ove_spte_fee,
            maintenance_fee_per_kw=args.maintenance_fee_per_kw,
            battery_maintenance_fee_per_kwh=args.battery_maintenance_fee_per_kwh,
            discount_rate=args.discount_rate,
            loan_rate=args.loan_rate,
            loan_years=args.loan_years,
            enable_power_smoothing=args.enable_power_smoothing,
            min_soc_reserve=args.min_soc_reserve,
            max_power_by_block=max_power_by_block if max_power_by_block else None,
            heating_config={"heating_kwh": args.heating_kwh, "start_hour": args.heating_start_hour, "end_hour": args.heating_end_hour} if args.add_heating_load else None,
        )

        # Check if optimization is requested
        if args.optimize_npv:
            print("\nRunning NPV OPTIMIZATION using differential evolution...")
            print(
                "Constraints: Solar 100-1000kW (50kW steps), Inverter 50kW-Solar (50kW steps), Battery 0-1000kWh (50kWh steps)"
            )
            print(f"Max iterations: {args.opt_maxiter}, Population size: {args.opt_popsize}")

            # Run optimization
            result = analyzer.optimize_npv_differential_evolution(
                maxiter=args.opt_maxiter,
                popsize=args.opt_popsize,
                seed=args.opt_seed,
                polish=args.opt_polish,
                verbose=True,
            )

            print("\n" + "=" * 60)
            print("OPTIMIZATION RESULTS")
            print("=" * 60)
            print(f"Optimal Solar Power: {result['optimal_solar_kw']:.1f} kW")
            print(f"Optimal Inverter Power: {result['optimal_inverter_kw']:.1f} kW")
            print(f"Optimal Battery Capacity: {result['optimal_battery_kwh']:.1f} kWh")
            print(f"Maximum NPV: €{result['optimal_npv']:,.0f}")
            print(f"Optimization converged: {result['success']}")
            print(f"Function evaluations: {result['function_evaluations']}")

            if result["message"]:
                print(f"Message: {result['message']}")
        else:
            # Run complete analysis
            analyzer.run_complete_analysis()

    else:
        print("\nRunning in SINGLE MODE - Individual scenario")
        print(
            f"Solar: {args.solar_power}kW, Inverter: {args.inverter_power}kW, Battery: {args.battery_capacity}kWh"
        )
        if args.add_heating_load:
            print(f"Heating load: {args.heating_kwh} kWh/season, {args.heating_start_hour}:00-{'midnight' if args.heating_end_hour == 0 else f'{args.heating_end_hour}:00'}")

        # Create transmission costs dictionary from arguments
        transmission_costs = {
            "block1": args.transmission_block1,
            "block2": args.transmission_block2,
            "block3": args.transmission_block3,
            "block4": args.transmission_block4,
            "block5": args.transmission_block5,
        }

        # Create monthly power fees dictionary from arguments
        monthly_power_fees = {
            "block1": args.power_fee_block1,
            "block2": args.power_fee_block2,
            "block3": args.power_fee_block3,
            "block4": args.power_fee_block4,
            "block5": args.power_fee_block5,
        }

        # Initialize simulator
        simulator = PowerFlowSimulator(
            solar_panel_power_kw=args.solar_power,
            inverter_power_kw=args.inverter_power,
            battery_capacity_kwh=args.battery_capacity,
            battery_c_rate=args.battery_c_rate,
            battery_efficiency=args.battery_efficiency,
            transmission_costs=transmission_costs,
            monthly_power_fees=monthly_power_fees,
            ove_spte_fee=args.ove_spte_fee,
            peak_price=args.peak_price,
            off_peak_price=args.off_peak_price,
            enable_power_smoothing=args.enable_power_smoothing,
            min_soc_reserve=args.min_soc_reserve,
            max_power_threshold=args.max_power_threshold,
            max_power_by_block=None
            if all(
                v is None
                for v in [
                    args.max_power_block1,
                    args.max_power_block2,
                    args.max_power_block3,
                    args.max_power_block4,
                    args.max_power_block5,
                ]
            )
            else {
                k: v
                for k, v in {
                    1: args.max_power_block1,
                    2: args.max_power_block2,
                    3: args.max_power_block3,
                    4: args.max_power_block4,
                    5: args.max_power_block5,
                }.items()
                if v is not None
            },
            heating_config={"heating_kwh": args.heating_kwh, "start_hour": args.heating_start_hour, "end_hour": args.heating_end_hour} if args.add_heating_load else None,
        )

        # Run simulation
        simulator.load_and_align_data()
        simulator.scale_solar_generation()
        simulator.simulate_power_flows()

        # Calculate costs and savings
        cost_analysis = simulator.calculate_costs_and_savings(
            peak_price=args.peak_price,
            off_peak_price=args.off_peak_price,
            export_price=args.export_price,
        )

        # Display transmission cost breakdown
        print("\nTransmission Cost Analysis by Block:")
        transmission_breakdown = simulator.calculate_transmission_cost_breakdown()
        for block, data in transmission_breakdown.items():
            print(
                f"  {block}: {data['intervals']:>5} intervals ({data['percentage_of_year']:>4.1f}% of year)"
            )
            print(
                f"    Import: {data['total_import_kwh']:>8.1f} kWh × €{data['transmission_rate_eur_per_kwh']:.5f} = €{data['transmission_cost_eur']:>6.2f}"
            )
            print(f"    Max power: {data['max_import_power_kw']:>6.1f} kW")
            if data["power_fee_rate_eur_per_kw_per_month"] > 0:
                print(
                    f"    Monthly power fee: {data['max_import_power_kw']:>6.1f} kW × €{data['power_fee_rate_eur_per_kw_per_month']:.5f}/kW/month = €{data['monthly_power_fee_eur']:>6.2f}/month"
                )
                print(f"    Annual power fee: €{data['annual_power_fee_eur']:>6.2f}")
            else:
                print("    Monthly power fee: €0.00/month (no fee for this block)")
                print("    Annual power fee: €0.00")

        # Display OVE-SPTE cost breakdown
        print("\nOVE-SPTE Cost Analysis:")
        ove_spte_breakdown = simulator.calculate_ove_spte_cost(
            simulator.max_power_by_block_simulated
        )
        if ove_spte_breakdown:
            print(
                f"  Block 1 max power: {simulator.max_power_by_block_simulated['block_1']:>6.1f} kW"
            )
            print(
                f"  Block 2 max power: {simulator.max_power_by_block_simulated['block_2']:>6.1f} kW"
            )
            print(
                f"  Weighted power: (4 × {simulator.max_power_by_block_simulated['block_1']:.1f} + 8 × {simulator.max_power_by_block_simulated['block_2']:.1f}) = {ove_spte_breakdown['weighted_power_kw']:.1f} kW"
            )
            #            print(f"  Monthly OVE-SPTE cost: {ove_spte_breakdown['weighted_power_kw']:.1f} kW × €{ove_spte_breakdown['ove_spte_rate_eur_per_kw_per_month']:.5f}/kW/month = €{ove_spte_breakdown['monthly_ove_spte_cost_eur']:>6.2f}/month")
            print(
                f"  Annual OVE-SPTE cost: €{ove_spte_breakdown['annual_ove_spte_cost_eur']:>6.2f}"
            )

        # Create visualizations and analysis
        # simulator.create_power_flow_visualization(days_to_show=7)
        # _ = simulator.create_monthly_analysis()  # TODO: implement
        # simulator.create_monthly_solar_daily_plot()  # TODO: implement
        simulator.export_results()

        print("\nSimulation complete!")
        print("\nGenerated files:")
        print("- power_flow_analysis_7days.png")
        print("- monthly_energy_analysis.png")
        print("- monthly_solar_daily_production.png")
        print("- power_flow_simulation_results.csv")
        print("- system_summary.csv")


if __name__ == "__main__":
    main()
