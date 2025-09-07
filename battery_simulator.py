#!/usr/bin/env python3
"""
Solar + Battery Storage Simulation
Analyzes cost savings for different battery capacities based on actual electricity usage data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
from datetime import datetime, time
import warnings
import argparse
warnings.filterwarnings('ignore')

class BatterySimulator:
    def __init__(self, data_file='data.csv', solar_scale_factor=1.0, enable_off_peak_charging=False, 
                 peak_price=0.23, off_peak_price=0.18, transmission_costs=None):
        """
        Initialize the battery simulator
        
        Args:
            data_file (str): Path to CSV file with electricity data
            solar_scale_factor (float): Factor to scale solar production (1.0 = current, 2.0 = double, etc.)
            enable_off_peak_charging (bool): Enable proactive charging during off-peak hours
            peak_price (float): Peak hour electricity price in EUR/kWh (6am-10pm weekdays)
            off_peak_price (float): Off-peak hour electricity price in EUR/kWh (nights and weekends)
            transmission_costs (dict): Transmission costs for 5 blocks × 2 seasons (default: all zeros)
        """
        self.data_file = data_file
        self.solar_scale_factor = solar_scale_factor
        self.enable_off_peak_charging = enable_off_peak_charging
        self.df = None
        self.results = {}
        self.solar_scaling_results = {}
        
        # Battery parameters
        self.efficiency = 1.0  # 100% round-trip efficiency (0 losses)
        self.max_charge_rate = 0.5  # 0.5C charging rate (can charge 50% of capacity per hour)
        self.max_discharge_rate = 0.5  # 0.5C discharge rate
        
        # Electricity pricing (EUR/kWh)
        self.peak_price = peak_price  # 6am-10pm weekdays
        self.off_peak_price = off_peak_price  # nights and weekends
        
        # Transmission costs (EUR/kWh) - 5 blocks (1=most expensive, 5=cheapest)
        if transmission_costs is None:
            transmission_costs = {
                'block1': 0.01998, 'block2': 0.01833, 'block3': 0.01809, 'block4': 0.01855, 'block5': 0.01873
            }
        self.transmission_costs = transmission_costs
        
    def load_and_preprocess_data(self):
        """Load CSV data and preprocess for simulation"""
        print("Loading and preprocessing data...")
        
        # Load CSV
        self.df = pd.read_csv(self.data_file)
        
        # Convert timestamp to datetime
        self.df['datetime'] = pd.to_datetime(self.df['Časovna značka'])
        
        # Extract relevant energy columns
        self.df['grid_import'] = self.df['Energija A+']  # kWh imported from grid
        self.df['solar_export_original'] = self.df['Energija A-']  # kWh exported to grid (original)
        
        # Scale solar export by the solar scale factor
        self.df['solar_export'] = self.df['solar_export_original'] * self.solar_scale_factor
        
        # Calculate net energy (positive = import, negative = export)
        self.df['net_energy'] = self.df['grid_import'] - self.df['solar_export']
        
        # Add time-based features
        self.df['hour'] = self.df['datetime'].dt.hour
        self.df['weekday'] = self.df['datetime'].dt.weekday  # 0=Monday, 6=Sunday
        self.df['month'] = self.df['datetime'].dt.month
        self.df['is_peak_hour'] = (
            (self.df['hour'] >= 6) & (self.df['hour'] < 22) & (self.df['weekday'] < 5)
        )
        
        # Determine transmission season (high season: Nov 1st to Mar 1st)
        self.df['is_high_season'] = (self.df['month'] >= 11) | (self.df['month'] <= 2)
        
        # Determine transmission time block based on complex rules
        def get_transmission_block(row):
            hour = row['hour']
            is_workday = row['weekday'] < 5  # Monday=0, Sunday=6
            is_high_season = row['is_high_season']
            
            if is_high_season:  # High season (Nov-Mar)
                if is_workday:  # High season workdays
                    if (7 <= hour < 14) or (16 <= hour < 20):  # 7-14h, 16-20h
                        return 1  # Most expensive
                    elif (6 <= hour < 7) or (14 <= hour < 16) or (20 <= hour < 22):  # 6-7h, 14-16h, 20-22h
                        return 2
                    else:  # midnight-6h, 22h-midnight
                        return 3
                else:  # High season non-workdays
                    # No 1st block (most expensive)
                    if (7 <= hour < 14) or (16 <= hour < 20):  # 7-14h, 16-20h
                        return 2
                    elif (6 <= hour < 7) or (14 <= hour < 16) or (20 <= hour < 22):  # 6-7h, 14-16h, 20-22h
                        return 3
                    else:  # midnight-6h, 22h-midnight
                        return 4
            else:  # Low season (Mar-Nov)
                if is_workday:  # Low season workdays
                    # No 1st block
                    if (7 <= hour < 14) or (16 <= hour < 20):  # 7-14h, 16-20h
                        return 2
                    elif (6 <= hour < 7) or (14 <= hour < 16) or (20 <= hour < 22):  # 6-7h, 14-16h, 20-22h
                        return 3
                    else:  # midnight-6h, 22h-midnight
                        return 4
                else:  # Low season non-workdays
                    # No 1st nor 2nd block
                    if (7 <= hour < 14) or (16 <= hour < 20):  # 7-14h, 16-20h
                        return 3
                    elif (6 <= hour < 7) or (14 <= hour < 16) or (20 <= hour < 22):  # 6-7h, 14-16h, 20-22h
                        return 4
                    else:  # midnight-6h, 22h-midnight
                        return 5
                        
        self.df['transmission_block'] = self.df.apply(get_transmission_block, axis=1)
        
        # Calculate transmission cost for each interval
        def get_transmission_cost(row):
            block_num = row['transmission_block']
            return self.transmission_costs[f'block{block_num}'] if f'block{block_num}' in self.transmission_costs else 0.0
            
        self.df['transmission_cost_per_kwh'] = self.df.apply(get_transmission_cost, axis=1)
        
        # Calculate total electricity price (base price + transmission cost)
        base_price = np.where(self.df['is_peak_hour'], self.peak_price, self.off_peak_price)
        self.df['price_per_kwh'] = base_price + self.df['transmission_cost_per_kwh']
        
        # Calculate baseline cost (without battery)
        self.df['baseline_cost'] = np.maximum(0, self.df['net_energy']) * self.df['price_per_kwh']
        
        print(f"Loaded {len(self.df)} data points from {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        print(f"Solar scale factor: {self.solar_scale_factor}x")
        print(f"Total baseline annual cost: €{self.df['baseline_cost'].sum():.2f}")
        
        # Show solar scaling impact
        original_export = self.df['solar_export_original'].sum()
        scaled_export = self.df['solar_export'].sum()
        print(f"Original solar export: {original_export:.2f} kWh/year")
        print(f"Scaled solar export: {scaled_export:.2f} kWh/year")
        
    def simulate_battery(self, capacity_kwh):
        """
        Simulate battery operation for given capacity
        
        Args:
            capacity_kwh (float): Battery capacity in kWh
            
        Returns:
            dict: Simulation results
        """
        # Initialize battery state
        soc = 0.0  # Start at 0% state of charge
        battery_log = []
        costs = []
        
        # Calculate max charge/discharge per 15-min interval
        max_charge_15min = capacity_kwh * self.max_charge_rate * 0.25  # 15min = 0.25 hours
        max_discharge_15min = capacity_kwh * self.max_discharge_rate * 0.25
        
        for _, row in self.df.iterrows():
            net_demand = row['net_energy']  # Positive = import needed, negative = export available
            price = row['price_per_kwh']
            is_peak_hour = row['is_peak_hour']
            
            # Initialize variables
            grid_import = 0
            off_peak_charging = 0
            
            # STEP 1: Handle solar surplus charging (highest priority - free energy)
            if net_demand < 0:  # Solar surplus available
                surplus = -net_demand
                remaining_capacity = capacity_kwh - soc
                charge_amount = min(surplus, max_charge_15min, remaining_capacity)
                charge_amount *= self.efficiency  # Account for charging losses
                soc += charge_amount
                # Export remaining surplus to grid (no cost)
                grid_import = max(0, net_demand + charge_amount/self.efficiency)
                
            # STEP 2: Handle demand with battery discharge (if needed)
            elif net_demand > 0:  # Need to import energy
                # Try to discharge battery first
                discharge_amount = min(net_demand, max_discharge_15min, soc)
                soc -= discharge_amount
                remaining_demand = net_demand - discharge_amount
                grid_import = max(0, remaining_demand)
            
            # STEP 3: Handle off-peak charging (lowest priority - cheap energy storage)
            if self.enable_off_peak_charging and not is_peak_hour:
                # Target SOC of 80% to leave room for solar surplus
                target_soc = capacity_kwh * 0.8
                
                if soc < target_soc:
                    # Calculate how much we want to charge
                    desired_charge = min(target_soc - soc, max_charge_15min)
                    actual_charge = desired_charge * self.efficiency
                    
                    # Add off-peak charging to grid import
                    off_peak_charging = desired_charge / self.efficiency
                    grid_import += off_peak_charging
                    soc += actual_charge
            
            # Ensure SOC stays within bounds
            soc = max(0, min(capacity_kwh, soc))
            
            # Calculate cost for this interval
            cost = grid_import * price
            costs.append(cost)
            
            battery_log.append({
                'datetime': row['datetime'],
                'net_demand': net_demand,
                'soc': soc,
                'soc_percent': (soc / capacity_kwh) * 100,
                'grid_import': grid_import,
                'cost': cost,
                'price': price
            })
        
        # Calculate results
        total_cost = sum(costs)
        baseline_cost = self.df['baseline_cost'].sum()
        savings = baseline_cost - total_cost
        savings_percent = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        # Battery utilization stats
        battery_df = pd.DataFrame(battery_log)
        avg_soc = battery_df['soc_percent'].mean()
        min_soc = battery_df['soc_percent'].min()
        max_soc = battery_df['soc_percent'].max()
        
        # Calculate approximate cycles per year (rough estimate)
        soc_changes = battery_df['soc'].diff().abs()
        total_throughput = soc_changes.sum()
        annual_cycles = total_throughput / (2 * capacity_kwh)  # Full cycle = capacity down + up
        
        return {
            'capacity_kwh': capacity_kwh,
            'total_cost': total_cost,
            'baseline_cost': baseline_cost,
            'annual_savings': savings,
            'savings_percent': savings_percent,
            'avg_soc_percent': avg_soc,
            'min_soc_percent': min_soc,
            'max_soc_percent': max_soc,
            'annual_cycles': annual_cycles,
            'battery_log': battery_df
        }
    
    def run_capacity_analysis(self, capacities=None):
        """Run simulation for multiple battery capacities"""
        if capacities is None:
            capacities = range(1, 21)  # 1-20 kWh
            
        print(f"\nRunning battery simulation for capacities: {list(capacities)} kWh")
        
        self.results = {}
        
        for capacity in capacities:
            print(f"  Simulating {capacity} kWh battery...")
            result = self.simulate_battery(capacity)
            self.results[capacity] = result
            
        print("\nSimulation complete!")
        
    def run_solar_scaling_analysis(self, solar_scales=None, capacities=None):
        """Run simulation for multiple solar scale factors and battery capacities"""
        if solar_scales is None:
            solar_scales = [1, 2, 3, 4, 5]  # 1x to 5x current solar capacity
        if capacities is None:
            capacities = range(1, 21)  # 1-20 kWh
            
        print(f"\nRunning solar scaling analysis...")
        print(f"Solar scales: {solar_scales}x")
        print(f"Battery capacities: {list(capacities)} kWh")
        
        self.solar_scaling_results = {}
        
        for scale in solar_scales:
            print(f"\n--- Solar Scale {scale}x ---")
            
            # Create a new simulator with this solar scale
            temp_simulator = BatterySimulator(self.data_file, solar_scale_factor=scale, enable_off_peak_charging=self.enable_off_peak_charging,
                                             peak_price=self.peak_price, off_peak_price=self.off_peak_price,
                                             transmission_costs=self.transmission_costs)
            temp_simulator.load_and_preprocess_data()
            temp_simulator.run_capacity_analysis(capacities)
            
            # Store results
            self.solar_scaling_results[scale] = {
                'baseline_cost': list(temp_simulator.results.values())[0]['baseline_cost'],
                'battery_results': temp_simulator.results
            }
        
        print("\nSolar scaling analysis complete!")
        
    def print_results_summary(self):
        """Print summary of results"""
        if not self.results:
            print("No results to display. Run simulation first.")
            return
            
        print("\n" + "="*80)
        print("BATTERY CAPACITY ANALYSIS RESULTS")
        print("="*80)
        
        baseline_cost = list(self.results.values())[0]['baseline_cost']
        print(f"Baseline annual cost (no battery): €{baseline_cost:.2f}")
        
        # Print current costs for each battery capacity
        print("\nCurrent costs with battery:")
        for capacity, result in sorted(self.results.items()):
            current_cost = result['total_cost']
            print(f"  {capacity} kWh battery: €{current_cost:.2f}")
        print()
        
        # Create summary table
        summary_data = []
        for capacity, result in sorted(self.results.items()):
            summary_data.append({
                'Capacity (kWh)': capacity,
                'Annual Cost (€)': f"{result['total_cost']:.2f}",
                'Annual Savings (€)': f"{result['annual_savings']:.2f}",
                'Savings (%)': f"{result['savings_percent']:.1f}%",
                'Avg SOC (%)': f"{result['avg_soc_percent']:.1f}%",
                'Annual Cycles': f"{result['annual_cycles']:.0f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Find optimal capacity
        best_capacity = max(self.results.keys(), key=lambda k: self.results[k]['annual_savings'])
        best_savings = self.results[best_capacity]['annual_savings']
        
        print(f"\nOPTIMAL BATTERY SIZE: {best_capacity} kWh")
        print(f"Maximum annual savings: €{best_savings:.2f}")
        print(f"Savings percentage: {self.results[best_capacity]['savings_percent']:.1f}%")
        
    def create_daily_soc_plot(self, capacities=None):
        """Create plot showing maximum daily state of charge for multiple battery capacities"""
        if capacities is None:
            capacities = [5, 10, 15, 20]  # Default capacities to compare
            
        # Filter to only include capacities we have results for
        available_capacities = [c for c in capacities if c in self.results]
        
        if not available_capacities:
            print(f"No results available for capacities {capacities}. Run simulation first.")
            return
        
        plt.figure(figsize=(16, 10))
        
        # Define colors for different capacities
        colors = plt.cm.viridis(np.linspace(0, 1, len(available_capacities)))
        
        for i, capacity_kwh in enumerate(available_capacities):
            battery_log = self.results[capacity_kwh]['battery_log']
            
            # Group by date and find max SOC for each day
            battery_log['date'] = battery_log['datetime'].dt.date
            daily_max_soc = battery_log.groupby('date')['soc_percent'].max().reset_index()
            
            # Plot the line for this capacity
            plt.plot(daily_max_soc['date'], daily_max_soc['soc_percent'], 
                     linewidth=2, color=colors[i], alpha=0.8, 
                     label=f'{capacity_kwh} kWh')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Maximum Daily State of Charge (%)', fontsize=12)
        plt.title('Daily Maximum Battery State of Charge - Multiple Battery Capacities', 
                  fontsize=14, fontweight='bold')
        
        # Format x-axis
        plt.xticks(rotation=45)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add horizontal reference lines
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add reference line labels
        plt.text(daily_max_soc['date'].iloc[-1], 102, 'Full Charge', 
                 fontsize=9, color='red', alpha=0.7)
        plt.text(daily_max_soc['date'].iloc[-1], 52, '50% Charge', 
                 fontsize=9, color='orange', alpha=0.7)
        plt.text(daily_max_soc['date'].iloc[-1], 2, 'Empty', 
                 fontsize=9, color='gray', alpha=0.7)
        
        plt.legend(title='Battery Capacity', loc='upper right')
        plt.tight_layout()
        
        # Save
        capacities_str = '_'.join(map(str, available_capacities))
        plt.savefig(f'daily_max_soc_comparison_{capacities_str}kwh.png', dpi=300, bbox_inches='tight')
        print(f"\nDaily SOC comparison plot saved as 'daily_max_soc_comparison_{capacities_str}kwh.png'")
        plt.close()
        
    def create_solar_scaling_plots(self, capacity_kwh=10):
        """Create plots comparing different solar scales for a specific battery capacity"""
        if not self.solar_scaling_results:
            print("No solar scaling results available. Run solar scaling analysis first.")
            return
            
        plt.figure(figsize=(16, 10))
        
        # Define colors for different solar scales
        solar_scales = sorted(self.solar_scaling_results.keys())
        colors = plt.cm.plasma(np.linspace(0, 1, len(solar_scales)))
        
        for i, scale in enumerate(solar_scales):
            if capacity_kwh in self.solar_scaling_results[scale]['battery_results']:
                battery_log = self.solar_scaling_results[scale]['battery_results'][capacity_kwh]['battery_log']
                
                # Group by date and find max SOC for each day
                battery_log['date'] = battery_log['datetime'].dt.date
                daily_max_soc = battery_log.groupby('date')['soc_percent'].max().reset_index()
                
                # Plot the line for this solar scale
                plt.plot(daily_max_soc['date'], daily_max_soc['soc_percent'], 
                         linewidth=2, color=colors[i], alpha=0.8, 
                         label=f'{scale}x Solar')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Maximum Daily State of Charge (%)', fontsize=12)
        plt.title(f'Daily Max SOC with Different Solar Capacities - {capacity_kwh} kWh Battery', 
                  fontsize=14, fontweight='bold')
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add horizontal reference lines
        plt.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=1)
        plt.axhline(y=50, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.legend(title='Solar Capacity', loc='upper right')
        plt.tight_layout()
        
        # Save
        scales_str = '_'.join(map(str, solar_scales))
        plt.savefig(f'solar_scaling_soc_{capacity_kwh}kwh_{scales_str}x.png', dpi=300, bbox_inches='tight')
        print(f"\nSolar scaling SOC plot saved as 'solar_scaling_soc_{capacity_kwh}kwh_{scales_str}x.png'")
        plt.close()
        
    def create_savings_heatmap(self):
        """Create heatmap showing savings across solar scales and battery capacities"""
        if not self.solar_scaling_results:
            print("No solar scaling results available. Run solar scaling analysis first.")
            return
            
        # Prepare data for heatmap
        solar_scales = sorted(self.solar_scaling_results.keys())
        capacities = sorted(list(self.solar_scaling_results[solar_scales[0]]['battery_results'].keys()))
        
        # Create savings matrix
        savings_matrix = []
        for scale in solar_scales:
            scale_savings = []
            for capacity in capacities:
                savings = self.solar_scaling_results[scale]['battery_results'][capacity]['annual_savings']
                scale_savings.append(savings)
            savings_matrix.append(scale_savings)
        
        # Create the heatmap
        plt.figure(figsize=(14, 8))
        
        savings_array = np.array(savings_matrix)
        im = plt.imshow(savings_array, cmap='viridis', aspect='auto')
        
        # Set ticks and labels
        plt.xticks(range(len(capacities)), [f'{c} kWh' for c in capacities])
        plt.yticks(range(len(solar_scales)), [f'{s}x' for s in solar_scales])
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Annual Savings (€)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(solar_scales)):
            for j in range(len(capacities)):
                text = f'€{savings_array[i, j]:.0f}'
                text_obj = plt.text(j, i, text, ha='center', va='center', 
                        color='white', fontsize=8, fontweight='bold')
                text_obj.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                          path_effects.Normal()])
        
        plt.xlabel('Battery Capacity')
        plt.ylabel('Solar Scale Factor')
        plt.title('Annual Savings Heatmap: Solar Scale vs Battery Capacity', 
                  fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('solar_battery_savings_heatmap.png', dpi=300, bbox_inches='tight')
        print("\nSavings heatmap saved as 'solar_battery_savings_heatmap.png'")
        plt.show()
        
    def create_roi_heatmap(self, solar_cost_per_factor=200, battery_cost_per_kwh=312.5):
        """Create heatmap showing ROI payback period in years"""
        if not self.solar_scaling_results:
            print("No solar scaling results available. Run solar scaling analysis first.")
            return
            
        # Prepare data for heatmap
        solar_scales = sorted(self.solar_scaling_results.keys())
        capacities = sorted(list(self.solar_scaling_results[solar_scales[0]]['battery_results'].keys()))
        
        # Get baseline savings (1x solar, 0 battery)
        baseline_1x_no_battery = self.solar_scaling_results[1]['baseline_cost']
        
        # Create ROI matrix
        roi_matrix = []
        for scale in solar_scales:
            scale_roi = []
            for capacity in capacities:
                # Calculate total investment cost
                solar_investment = (scale - 1) * solar_cost_per_factor  # Additional solar beyond 1x
                battery_investment = capacity * battery_cost_per_kwh
                total_investment = solar_investment + battery_investment
                
                # Calculate annual savings compared to baseline (1x solar, no battery)
                current_cost = self.solar_scaling_results[scale]['battery_results'][capacity]['total_cost']
                annual_savings = baseline_1x_no_battery - current_cost
                
                # Calculate ROI payback period in years
                if annual_savings > 0 and total_investment > 0:
                    payback_years = total_investment / annual_savings
                    # Cap at 10 years for display purposes
                    payback_years = min(payback_years, 10)
                elif total_investment == 0:
                    payback_years = 0  # No investment needed
                else:
                    payback_years = 10  # No savings or negative ROI
                    
                scale_roi.append(payback_years)
            roi_matrix.append(scale_roi)
        
        # Create the heatmap
        plt.figure(figsize=(14, 8))
        
        roi_array = np.array(roi_matrix)
        
        # Use a custom colormap where green = good ROI, red = poor ROI
        im = plt.imshow(roi_array, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=10)
        
        # Set ticks and labels
        plt.xticks(range(len(capacities)), [f'{c} kWh' for c in capacities])
        plt.yticks(range(len(solar_scales)), [f'{s}x' for s in solar_scales])
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Payback Period (Years)', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(solar_scales)):
            for j in range(len(capacities)):
                years = roi_array[i, j]
                if years >= 10:
                    text = '>10'
                elif years == 0:
                    text = '0'
                else:
                    text = f'{years:.1f}'
                    
                text_obj = plt.text(j, i, text, ha='center', va='center', 
                        color='white', fontsize=8, fontweight='bold')
                text_obj.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                          path_effects.Normal()])
        
        plt.xlabel('Battery Capacity')
        plt.ylabel('Solar Scale Factor')
        plt.title(f'ROI Payback Period Heatmap (Solar: €{solar_cost_per_factor}/factor, Battery: €{battery_cost_per_kwh}/kWh)', 
                  fontsize=12, fontweight='bold')
        
        # Add investment cost annotations
        plt.figtext(0.02, 0.02, f'Investment costs:\n• Solar: €{solar_cost_per_factor} per additional factor\n• Battery: €{battery_cost_per_kwh} per kWh', 
                    fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('solar_battery_roi_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"\nROI payback heatmap saved as 'solar_battery_roi_heatmap.png'")
        plt.show()
        
        # Print some key insights
        print(f"\nROI Analysis Summary (Solar: €{solar_cost_per_factor}/factor, Battery: €{battery_cost_per_kwh}/kWh):")
        
        # Find best ROI combinations (payback < 10 years)
        good_roi_combinations = []
        for i, scale in enumerate(solar_scales):
            for j, capacity in enumerate(capacities):
                if roi_array[i, j] < 10 and roi_array[i, j] > 0:
                    good_roi_combinations.append((scale, capacity, roi_array[i, j]))
        
        if good_roi_combinations:
            print("Best ROI combinations (payback < 10 years):")
            good_roi_combinations.sort(key=lambda x: x[2])  # Sort by payback period
            for scale, capacity, payback in good_roi_combinations[:5]:  # Show top 5
                investment = (scale - 1) * solar_cost_per_factor + capacity * battery_cost_per_kwh
                print(f"  {scale}x solar + {capacity} kWh battery: {payback:.1f} years (€{investment} investment)")
        else:
            print("No combinations with payback < 10 years found with current pricing.")
        
    def export_detailed_results(self):
        """Export detailed results to CSV"""
        if not self.results:
            print("No results to export. Run simulation first.")
            return
            
        # Create summary CSV
        summary_data = []
        for capacity, result in sorted(self.results.items()):
            summary_data.append({
                'capacity_kwh': capacity,
                'baseline_cost_eur': result['baseline_cost'],
                'total_cost_eur': result['total_cost'],
                'annual_savings_eur': result['annual_savings'],
                'savings_percent': result['savings_percent'],
                'avg_soc_percent': result['avg_soc_percent'],
                'min_soc_percent': result['min_soc_percent'],
                'max_soc_percent': result['max_soc_percent'],
                'annual_cycles': result['annual_cycles']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('battery_capacity_analysis.csv', index=False)
        print("Summary results exported to 'battery_capacity_analysis.csv'")
        
        # Export detailed battery operation for best capacity
        best_capacity = max(self.results.keys(), key=lambda k: self.results[k]['annual_savings'])
        best_result = self.results[best_capacity]
        
        detailed_df = best_result['battery_log'].copy()
        detailed_df.to_csv(f'battery_operation_{best_capacity}kwh.csv', index=False)
        print(f"Detailed battery operation (best case) exported to 'battery_operation_{best_capacity}kwh.csv'")


def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Solar Battery Storage Simulator with Solar Scaling Analysis")
    parser.add_argument('--off-peak-charging', action='store_true',
                        help='Enable proactive battery charging during off-peak hours (weekends and weekday nights)')
    parser.add_argument('--peak-price', type=float, default=0.23,
                        help='Peak hour electricity price in EUR/kWh (6am-10pm weekdays, default: 0.23)')
    parser.add_argument('--off-peak-price', type=float, default=0.18,
                        help='Off-peak hour electricity price in EUR/kWh (nights and weekends, default: 0.18)')
    
    # Transmission cost arguments (5 blocks, 1=most expensive, 5=cheapest)
    parser.add_argument('--transmission-block1', type=float, default=0.01998,
                        help='Block 1 transmission cost in EUR/kWh (most expensive, only high season workdays, default: 0.01998)')
    parser.add_argument('--transmission-block2', type=float, default=0.01833,
                        help='Block 2 transmission cost in EUR/kWh (high season all days, low season workdays, default: 0.01833)')
    parser.add_argument('--transmission-block3', type=float, default=0.01809,
                        help='Block 3 transmission cost in EUR/kWh (default: 0.01809)')
    parser.add_argument('--transmission-block4', type=float, default=0.01855,
                        help='Block 4 transmission cost in EUR/kWh (default: 0.01855)')
    parser.add_argument('--transmission-block5', type=float, default=0.01873,
                        help='Block 5 transmission cost in EUR/kWh (cheapest, only low season non-workdays, default: 0.01873)')
    args = parser.parse_args()
    
    # Validate price arguments
    if args.peak_price <= 0:
        parser.error("Peak price must be a positive number")
    if args.off_peak_price <= 0:
        parser.error("Off-peak price must be a positive number")
    
    # Validate transmission cost arguments (must be non-negative)
    transmission_costs = [
        args.transmission_block1, args.transmission_block2, args.transmission_block3,
        args.transmission_block4, args.transmission_block5
    ]
    for i, cost in enumerate(transmission_costs, 1):
        if cost < 0:
            parser.error(f"Transmission block {i} cost must be non-negative")
    
    print("Solar Battery Storage Simulator with Solar Scaling Analysis")
    print("="*60)
    
    # Show pricing information
    print(f"Peak hour price: €{args.peak_price:.3f}/kWh (6am-10pm weekdays)")
    print(f"Off-peak price: €{args.off_peak_price:.3f}/kWh (nights and weekends)")
    
    if args.off_peak_charging:
        print("Off-peak charging: ENABLED")
        print(f"Battery will charge proactively during off-peak hours (€{args.off_peak_price:.3f}/kWh)")
    else:
        print("Off-peak charging: DISABLED")
        print("Battery will only charge from solar surplus")
    print()
    
    # Create transmission costs dictionary from arguments
    transmission_costs = {
        'block1': args.transmission_block1, 'block2': args.transmission_block2,
        'block3': args.transmission_block3, 'block4': args.transmission_block4,
        'block5': args.transmission_block5
    }
    
    # Initialize simulator with current solar capacity (1x)
    simulator = BatterySimulator('data.csv', solar_scale_factor=1.0, enable_off_peak_charging=args.off_peak_charging,
                                 peak_price=args.peak_price, off_peak_price=args.off_peak_price,
                                 transmission_costs=transmission_costs)
    
    # Load and preprocess data
    simulator.load_and_preprocess_data()
    
    # Run basic simulation for 1-20 kWh capacities
    simulator.run_capacity_analysis([1.6, 3.2, 4.8])
    
    # Display results for current solar setup
    simulator.print_results_summary()
    
    # Create daily SOC comparison plot for multiple battery capacities
    simulator.create_daily_soc_plot([1.6, 3.2, 4.8])
    
    # Run solar scaling analysis (1x to 5x solar capacity)
    print("\n" + "="*60)
    print("SOLAR SCALING ANALYSIS")
    print("="*60)
    
    simulator.run_solar_scaling_analysis(
        solar_scales=[1, 2, 3, 4, 5], 
        capacities=[1.6, 3.2, 4.8]
    )
    
    # Create solar scaling visualizations
    simulator.create_solar_scaling_plots(capacity_kwh=5)  # 5 kWh battery example
    simulator.create_savings_heatmap()
    simulator.create_roi_heatmap(solar_cost_per_factor=240, battery_cost_per_kwh=312.5)
    
    # Export results
    simulator.export_detailed_results()
    
    print("\nAnalysis complete! Check the generated files for detailed results.")
    print("\nGenerated files:")
    print("- daily_max_soc_comparison_1_2_3_4_5_6_7_8kwh.png")
    print("- solar_scaling_soc_5kwh_1_2_3_4_5x.png") 
    print("- solar_battery_savings_heatmap.png")
    print("- solar_battery_roi_heatmap.png")
    print("- battery_capacity_analysis.csv")
    print("- battery_operation_*kwh.csv")


if __name__ == "__main__":
    main()
