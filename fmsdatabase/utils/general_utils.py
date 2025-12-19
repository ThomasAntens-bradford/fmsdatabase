from enum import Enum, auto

import numpy as np
import re
from scipy.stats import ttest_ind
import json
import os
from IPython.display import display
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from pandas.io.formats.style import Styler
import ipyvuetify as v
import pandas as pd
from typing import Any

class GroupedBy(Enum):
    LPT_PRESSURE = 'lpt_pressure'
    LPT_VOLTAGE = 'lpt_voltage'

class TVTestParameters(Enum):
    LOGTIME = 'logtime'
    ANODE_FLOW = 'anode_flow_rate'
    GAS_SELECT = 'gas_select'
    FILTERED_OUTLET_TEMP = 'filtered_outlet_temp'
    BODY_TEMP = 'body_temp'
    OUTLET_TEMP = 'outlet_temp'
    FILTERED_BODY_TEMP = 'filtered_body_temp'

class FRParts(Enum):
    FILTER = 'ejay filter'
    OUTLET = 'restrictor outlet'
    RESTRICTOR = 'flow restrictor'
    ANODE = 'anode flow restrictor'
    CATHODE = 'cathode flow restrictor'

class TVParts(Enum):
    MAIN_BODY = 'thermal valve main body'
    PLUNGER = 'thermal valve plunger'
    NUT = 'thermal valve nut'
    SEALING = 'thermal valve sealing element'
    GASKET = 'thermal valve gasket'
    HOLDER_1 = 'thermal valve holder 1'
    HOLDER_2 = 'thermal valve holder 2'
    WELD = 'thermal valve weld only'

class LimitStatus(Enum):
    """
    Enum representing within limits (True), outside limits (False), or on limit (On Limit).
    """
    TRUE = 'true'
    FALSE = 'false'
    ON_LIMIT = 'on_limit'

class FunctionalTestType(Enum):
    HIGH_OPEN_LOOP = 'high_open_loop'
    HIGH_CLOSED_LOOP = 'high_closed_loop'
    LOW_OPEN_LOOP = 'low_open_loop'
    LOW_CLOSED_LOOP = 'low_closed_loop'
    LOW_SLOPE = 'low_slope'
    HIGH_SLOPE = 'high_slope'
    ROOM = 'room_temp'
    HOT = 'hot_temp'
    COLD = 'cold_temp'
    NONE = 'none'

class FMSParts(Enum):
    MANIFOLD = 'manifold'
    TV = 'thermal valve'
    HPIV = 'hpiv'
    
class FMSProgressStatus(Enum):
    AWAITING_PART_AVAILABILITY = 'awaiting_part_availability'
    ASSEMBLY_COMPLETED = 'assembly_completed'
    TESTING_COMPLETED = 'testing_completed'
    TESTING = 'testing'
    TVAC_COMPLETED = 'tvac_completed'
    READY_FOR_TVAC = 'ready_for_tvac'
    DELIVERED = 'delivered'
    SHIPMENT = 'shipment'
    SCRAPPED = 'scrapped'

class FRStatus(Enum):
    DIFF_THICKNESS = 'diff_thickness'
    DIFF_ORIFICE = 'diff_orifice'
    DIFF_FLOWRATE = 'diff_flow_rate'
    DIFF_GEOMETRY = 'diff_geometry'
    DIFF_RADIUS = 'diff_radius'
    OK = 'ok'

class TVProgressStatus(Enum):
    """
    Enum representing the status of a process.
    """
    COMPLETED = 'completed'
    ASSEMBLY_COMPLETED = 'assembly_completed'
    SETTING_TEMPERATURE = 'setting_temperature'
    WELDING_TEST = 'welding_test'
    READY_FOR_WELD = 'ready_for_weld'
    WELDING_COMPLETED = 'welding_completed'
    TESTING_COMPLETED = 'testing_completed'
    COIL_MOUNTED = 'coil_mounted'
    FAILED = 'failed'

class ManifoldProgressStatus(Enum):
    FLOW_TESTING = 'flow_testing'
    AC_RATIO_SET = 'ac_ratio_set'
    WELDING_COMPLETED = 'welding_completed'
    COMPLETED = 'completed'
    AVAILABLE = 'available'
    ASSEMBLY_COMPLETED = 'assembly_completed'

class FMSFlowTestParameters(Enum):
    LOGTIME = 'logtime'
    Tu = 'Tu'
    Ku = 'Ku'
    HEATER_GAIN = 'heater_gain'
    HEATER_INTEGRAL = 'heater_integral'
    CLOSED_LOOP_TEMP = 'closed_loop_temp'
    LPT_VOLTAGE = 'lpt_voltage'
    LPT_PRESSURE = 'lpt_pressure'
    BRIDGE_VOLTAGE = 'bridge_voltage'
    LPT_TEMP = 'lpt_temp'
    DUTY_CYCLE_2 = 'duty_cycle_2'
    DUTY_CYCLE = 'duty_cycle'
    CLOSED_LOOP_PRESSURE = 'closed_loop_pressure'
    INLET_PRESSURE = 'inlet_pressure'
    PC1_PRESSURE = 'pc1_pressure'
    PC1_SETPOINT = 'pc1_setpoint'
    PC3_PRESSURE = 'pc3_pressure'
    PC3_SETPOINT = 'pc3_setpoint'
    ANODE_PRESSURE = 'anode_pressure'
    ANODE_TEMP = 'anode_temp'
    ANODE_FLOW = 'anode_flow'
    CATHODE_PRESSURE = 'cathode_pressure'
    CATHODE_TEMP = 'cathode_temp'
    CATHODE_FLOW = 'cathode_flow'
    ANODE_CATHODE_RATIO = 'anode_cathode_ratio'
    VACUUM_PRESSURE = 'vacuum_pressure'
    TV_PT1000 = 'tv_pt1000'
    ANODE_EST_FLOW = 'anode_est_flow'
    CATHODE_EST_FLOW = 'cathode_est_flow'
    AC_GAS_SELECT = 'ac_gas_select'
    FILTERED_LPT_TEMP = 'filtered_lpt_temp'
    HPIV_STATUS = 'hpiv_status'
    TV_POWER = 'tv_power'
    TV_VOLTAGE = 'tv_voltage'
    TV_CURRENT = 'tv_current'
    TOTAL_FLOW = 'total_flow'
    AVG_TV_POWER = 'avg_tv_power'

class FMSMainParameters(Enum):
    SERIAL_NUMBER = 'serial_number'
    MASS = 'mass'
    POWER_BUDGET_COLD = 'power_budget_cold'
    POWER_BUDGET_ROOM = 'power_budget_room'
    POWER_BUDGET_HOT = 'power_budget_hot'
    ROOM_HPIV_DROPOUT_VOLTAGE = 'room_hpiv_dropout_voltage'
    ROOM_HPIV_PULLIN_VOLTAGE = 'room_hpiv_pullin_voltage'
    ROOM_HPIV_CLOSING_RESPONSE = 'room_hpiv_closing_response'
    ROOM_HPIV_HOLD_POWER = 'room_hpiv_hold_power'
    ROOM_HPIV_OPENING_RESPONSE = 'room_hpiv_opening_response'
    ROOM_HPIV_OPENING_POWER = 'room_hpiv_opening_power'
    ROOM_HPIV_INDUCTANCE = 'room_hpiv_inductance'
    ROOM_TV_INDUCTANCE = 'room_tv_inductance'
    ROOM_HPIV_RESISTANCE = 'room_hpiv_resistance'
    ROOM_TVPT_RESISTANCE = 'room_tvpt_resistance'
    ROOM_TV_RESISTANCE = 'room_tv_resistance'
    ROOM_LPT_RESISTANCE = 'room_lpt_resistance'
    ROOM_TV_HIGH_LEAK = 'room_tv_high_leak'
    ROOM_TV_LOW_LEAK = 'room_tv_low_leak'
    ROOM_HPIV_HIGH_LEAK = 'room_hpiv_high_leak'
    ROOM_HPIV_LOW_LEAK = 'room_hpiv_low_leak'
    COLD_HPIV_DROPOUT_VOLTAGE = 'cold_hpiv_dropout_voltage'
    COLD_HPIV_PULLIN_VOLTAGE = 'cold_hpiv_pullin_voltage'
    COLD_HPIV_CLOSING_RESPONSE = 'cold_hpiv_closing_response'
    COLD_HPIV_HOLD_POWER = 'cold_hpiv_hold_power'
    COLD_HPIV_OPENING_RESPONSE = 'cold_hpiv_opening_response'
    COLD_HPIV_OPENING_POWER = 'cold_hpiv_opening_power'
    COLD_HPIV_INDUCTANCE = 'cold_hpiv_inductance'
    COLD_TV_INDUCTANCE = 'cold_tv_inductance'
    COLD_HPIV_RESISTANCE = 'cold_hpiv_resistance'
    COLD_TVPT_RESISTANCE = 'cold_tvpt_resistance'
    COLD_TV_RESISTANCE = 'cold_tv_resistance'
    COLD_LPT_RESISTANCE = 'cold_lpt_resistance'
    COLD_TV_HIGH_LEAK = 'cold_tv_high_leak'
    COLD_TV_LOW_LEAK = 'cold_tv_low_leak'
    COLD_HPIV_HIGH_LEAK = 'cold_hpiv_high_leak'
    COLD_HPIV_LOW_LEAK = 'cold_hpiv_low_leak'
    HOT_HPIV_DROPOUT_VOLTAGE = 'hot_hpiv_dropout_voltage'
    HOT_HPIV_PULLIN_VOLTAGE = 'hot_hpiv_pullin_voltage'
    HOT_HPIV_CLOSING_RESPONSE = 'hot_hpiv_closing_response'
    HOT_HPIV_HOLD_POWER = 'hot_hpiv_hold_power'
    HOT_HPIV_OPENING_RESPONSE = 'hot_hpiv_opening_response'
    HOT_HPIV_OPENING_POWER = 'hot_hpiv_opening_power'
    HOT_HPIV_INDUCTANCE = 'hot_hpiv_inductance'
    HOT_TV_INDUCTANCE = 'hot_tv_inductance'
    HOT_HPIV_RESISTANCE = 'hot_hpiv_resistance'
    HOT_TVPT_RESISTANCE = 'hot_tvpt_resistance'
    HOT_TV_RESISTANCE = 'hot_tv_resistance'
    HOT_LPT_RESISTANCE = 'hot_lpt_resistance'
    HOT_TV_HIGH_LEAK = 'hot_tv_high_leak'
    HOT_TV_LOW_LEAK = 'hot_tv_low_leak'
    HOT_HPIV_HIGH_LEAK = 'hot_hpiv_high_leak'
    HOT_HPIV_LOW_LEAK = 'hot_hpiv_low_leak'
    TV_HIGH_LEAK = 'tv_high_leak'
    TV_LOW_LEAK = 'tv_low_leak'
    HPIV_HIGH_LEAK = 'hpiv_high_leak'
    HPIV_LOW_LEAK = 'hpiv_low_leak'
    INLET_LOCATION = 'inlet_location'
    OUTLET_ANODE = 'outlet_anode'
    OUTLET_CATHODE = 'outlet_cathode'
    FMS_ENVELOPE = 'fms_envelope'
    TV_HOUSING_BONDING = 'tv_housing_bonding'
    BONDING_TV_HOUSING = 'bonding_tv_housing'
    TV_HOUSING_HPIV = 'tv_housing_hpiv'
    HPIV_HOUSING_TV = 'hpiv_housing_tv'
    LPT_HOUSING_BONDING = 'lpt_housing_bonding'
    BONDING_LPT_HOUSING = 'bonding_lpt_housing'
    J01_BONDING = 'j01_bonding'
    BONDING_J01 = 'bonding_j01'
    J02_BONDING = 'j02_bonding'
    BONDING_J02 = 'bonding_j02'
    J01_PIN_BONDING = 'j01_pin_bonding'
    BONDING_J01_PIN = 'bonding_j01_pin'
    J02_PIN_BONDING = 'j02_pin_bonding'
    BONDING_J02_PIN = 'bonding_j02_pin'
    LPT_PSIG = 'lpt_psig'
    LPT_PSIGRTN = 'lpt_psig_rtn'
    ISO_LPT_TSIG = 'iso_lpt_tsig'
    ISO_LPT_TSIGRTN = 'iso_lpt_tsig_rtn'
    LPT_PWR = 'lpt_power'
    LPT_PWRRTN = 'lpt_power_rtn'
    ISO_PT_SGN = 'iso_pt_sgn'
    ISO_PT_SGNRTN = 'iso_pt_sgn_rtn'
    TV_PWR = 'tv_power'
    TV_PWRRTN = 'tv_power_rtn'
    HPIV_PWR = 'hpiv_power'
    HPIV_PWRRTN = 'hpiv_power_rtn'
    CAP_LPT_TSIG = 'cap_lpt_tsig'
    CAP_LPT_TSIGRTN = 'cap_lpt_tsig_rtn'
    CAP_PT_SGN = 'cap_pt_sgn'
    CAP_PT_SGNRTN = 'cap_pt_sgn_rtn'
    LPT_T_RESISTANCE = 'lpt_t_resistance'
    TV_RESISTANCE = 'tv_resistance'
    TV_PT_RESISTANCE = 'tv_pt_resistance'
    HPIV_RESISTANCE = 'hpiv_resistance'
    HPIV_OPENING_POWER = 'hpiv_opening_power'
    HPIV_OPENING_RESPONSE = 'hpiv_opening_response'
    HPIV_HOLD_POWER = 'hpiv_hold_power'
    HPIV_CLOSING_RESPONSE = 'hpiv_closing_response'
    HPIV_PULLIN_VOLTAGE = 'hpiv_pullin_voltage'
    HPIV_DROPOUT_VOLTAGE = 'hpiv_dropout_voltage'
    LOW_PRESSURE_EXT_LEAK = 'low_pressure_ext_leak'
    HIGH_PRESSURE_EXT_LEAK_LOW = 'high_pressure_ext_leak_low'
    HIGH_PRESSURE_EXT_LEAK_HIGH = 'high_pressure_ext_leak_high'

class FMSTvacParameters(Enum):
    TIME = 'logtime'
    TRP1 = 'trp1'
    TRP2 = 'trp2'
    TV_INLET_TEMP = 'tv_inlet'
    MANIFOLD_TEMP = 'manifold'
    LPT_TEMP = 'lpt'
    HPIV_TEMP = 'hpiv'
    TV_OUTLET_TEMP = 'tv_outlet'
    FMS_INLET_TEMP = 'fms_inlet'
    ANODE_OUTLET_TEMP = 'anode'
    CATHODE_OUTLET_TEMP = 'cathode'

class TVTvacParameters(Enum):
    SCAN = "scan"
    TIME = "time"
    OUTLET_TEMP_1 = "outlet_temp_1"
    ALARM_101 = "alarm_101"
    OUTLET_TEMP_2 = "outlet_temp_2"
    ALARM_102 = "alarm_102"
    INTERFACE_TEMP = "interface_temp"
    ALARM_103 = "alarm_103"
    IF_PLATE = "if_plate"
    ALARM_104 = "alarm_104"
    VACUUM = "vacuum"
    ALARM_109 = "alarm_109"
    TV_VOLTAGE = "tv_voltage"
    ALARM_110 = "alarm_110"
    TV_CURRENT = "tv_current"
    ALARM_121 = "alarm_121"

class TVTvacParameters2(Enum):
    SCAN = "scan"
    TIME = "time"
    OUTLET_ELBOW = "outlet_elbow"
    ALARM_101 = "alarm_101"
    OUTLET_TEMP_1 = "outlet_temp_1"
    ALARM_102 = "alarm_102"
    INTERFACE_TEMP = "interface_temp"
    ALARM_103 = "alarm_103"
    OUTLET_TEMP_2 = "outlet_temp_2"
    ALARM_104 = "alarm_104"
    IF_PLATE_1 = "if_plate_1"
    ALARM_105 = "alarm_105"
    IF_PLATE_2 = "if_plate_2"
    ALARM_106 = "alarm_106"
    VACUUM = "vacuum"
    ALARM_109 = "alarm_109"
    TV_VOLTAGE = "tv_voltage"
    ALARM_110 = "alarm_110"
    TV_CURRENT = "tv_current"
    ALARM_121 = "alarm_121"

class HPIVParameters(Enum):
    """
    Enum defining all HPIV (High Pressure Isolation Valve) test parameters.
    
    This enumeration contains all the parameters that can be measured or tested
    during HPIV acceptance testing, including pressure tests, electrical tests,
    vibration tests, and cleanliness measurements.
    """
    HPIV_ID = "serial_nr"
    WEIGHT = "weight"
    PROOF_CLOSED = "proof_closed"
    PROOF_OPEN = "proof_open"
    LEAK_4_HP = "leak_4_hp"
    LEAK_4_LP = "leak_4_lp"
    LEAK_6_HP = "leak_6_hp"
    LEAK_6_LP = "leak_6_lp"
    LEAK_15_HP = "leak_15_hp"
    LEAK_15_LP = "leak_15_lp"
    LEAK_4_HP_PRESS = "leak_4_hp_press"
    LEAK_4_LP_PRESS = "leak_4_lp_press"
    LEAK_6_HP_PRESS = "leak_6_hp_press"
    LEAK_6_LP_PRESS = "leak_6_lp_press"
    LEAK_15_HP_PRESS = "leak_15_hp_press"
    LEAK_15_LP_PRESS = "leak_15_lp_press"
    DIELECTRIC_STR = "dielectric_str"
    INSULATION_RES = "insulation_res"
    POWER_TEMP = "power_temp"
    POWER_RES = "power_res"
    POWER_POWER = "power_power"
    EXT_LEAK = "ext_leak"
    PULLIN_PRES = "pullin_pres"
    PULLIN_VOLT = "pullin_volt"
    DROPOUT_VOLT = "dropout_volt"
    RESPO_PRES = "resp_pres"
    RESPO_VOLT = "respo_volt"
    RESPO_TIME = "respo_time"
    RESPC_VOLT = "respc_volt"
    RESPC_TIME = "respc_time"
    FLOWRATE = "flowrate"
    PRESSD = "pressd"
    CLEANLINESS_6_10 = "cleanliness_6_10"
    CLEANLINESS_11_25 = "cleanliness_11_25"
    CLEANLINESS_26_50 = "cleanliness_26_50"
    CLEANLINESS_51_100 = "cleanliness_51_100"
    CLEANLINESS_100 = "cleanliness_100"
    BEFORE_VIB_PEAK_X = "before_vib_peak_x"
    BEFORE_VIB_FREQ_X = "before_vib_freq_x"
    BEFORE_VIB_PEAK_Y = "before_vib_peak_y"
    BEFORE_VIB_FREQ_Y = "before_vib_freq_y"
    AFTER_VIB_PEAK_X = "after_vib_peak_x"
    AFTER_VIB_FREQ_X = "after_vib_freq_x"
    AFTER_VIB_PEAK_Y = "after_vib_peak_y"
    AFTER_VIB_FREQ_Y = "after_vib_freq_y"
    VIB_GRMS_X = "vib_grms_x"
    VIB_GRMS_Y = "vib_grms_y"

class HPIVParts(Enum):
    HPIV = "hpiv"
    SEAT_BODY = "seat body"
    COIL_ASSY = "coil assy"
    SPOOL_ASSY = "spool assy"
    SPOOL_ASSY_R = "spool assy_r"
    LOWER_SPOOL = "lower spool"
    NON_MAGNETIC_TUBE = "non magnetic tube"
    UPPER_SPOOL = "upper spool"
    COPPER_WIRE = "copper wire"
    KAPTON_TAPE = "kapton tape"
    LEAD_WIRE = "lead wire"
    SHRINK_TUBE = "shrink tube"
    SOLDER_FILLER = "solder filler"
    HOUSING = "housing"
    PLUNGER_ASSY = "plunger assy"
    PLUNGER_R = "plunger_r"
    SEAL = "seal"
    DISK_SPRING = "disk spring"
    SPRING = "spring"
    SHIM1 = "shim1"
    SHIM2 = "shim2"
    FILTER_ASSY = "filter assy"
    FRAME = "frame"
    SUPPORTER = "supporter"
    MESH = "mesh"

class LPTCoefficientParameters(Enum):
    LPT_ID = 'lpt_id'
    A0_P = 'a0_p'
    A1_P = 'a1_p'
    A2_P = 'a2_p'
    A3_P = 'a3_p'
    B0_P = 'b0_p'
    B1_P = 'b1_p'
    B2_P = 'b2_p'
    B3_P = 'b3_p'
    C0_P = 'c0_p'
    C1_P = 'c1_p'
    C2_P = 'c2_p'
    C3_P = 'c3_p'
    D0_P = 'd0_p'
    D1_P = 'd1_p'
    D2_P = 'd2_p'
    D3_P = 'd3_p'
    A0_T = 'a0_t'
    A1_T = 'a1_t'
    A2_T = 'a2_t'
    A3_T = 'a3_t'
    B0_T = 'b0_t'
    B1_T = 'b1_t'
    B2_T = 'b2_t'
    B3_T = 'b3_t'
    C0_T = 'c0_t'
    C1_T = 'c1_t'
    C2_T = 'c2_t'
    C3_T = 'c3_t'
    D0_T = 'd0_t'
    D1_T = 'd1_t'
    D2_T = 'd2_t'
    D3_T = 'd3_t'

def close_stats(d1: list[float], d2: list[float], rtol: float = 1e-1, atol: float = 1e-1) -> bool:
    mean_close = abs(np.mean(d1) - np.mean(d2)) <= max(rtol * max(abs(np.mean(d1)), abs(np.mean(d2))), atol)
    std_close = abs(np.std(d1) - np.std(d2)) <= max(rtol * max(abs(np.std(d1)), abs(np.std(d2))), atol)
    return mean_close and std_close

def compare_distributions(data1: list[float], data2: list[float], alpha=0.05) -> bool | None:
    """
    Compares two distributions for significant difference.
    
    Returns:
        True  → distributions are similar
        False → significant difference found
        None  → insufficient data
    """
    # t_stat, p_value = ttest_ind(data1, data2, equal_var=True)

    # if p_value < alpha:
    #     print(p_value, alpha)
    #     return False
    
    if close_stats(data1, data2):  # Pass x to close_stats
        return True

    return False

def compare_value_to_distribution(value: float, distribution: list[float], rtol: float = 1e-4, atol: float = 1e-8) -> LimitStatus:
    """
    Compare a single value to a distribution and return LimitStatus.

    - Returns TRUE if the value is clearly within mean ± std (with tolerances)
    - Returns ON_LIMIT if it's near the edge (within tolerance range)
    - Returns FALSE if it's outside the expected range
    """
    if len(distribution) == 0:
        raise ValueError("Distribution must not be empty")

    mean = np.mean(distribution)
    std = np.std(distribution)
    threshold = max(rtol * max(abs(mean), abs(value)), atol)

    lower_bound = mean - std
    upper_bound = mean + std

    if lower_bound + threshold < value < upper_bound - threshold:
        return LimitStatus.TRUE
    elif abs(value - lower_bound) <= threshold or abs(value - upper_bound) <= threshold:
        return LimitStatus.ON_LIMIT
    else:
        return LimitStatus.FALSE

def get_chunk_size(n: int, max_chunk: int | None = None) -> int:
    """
    Return a divisor of n to use as chunk size.
    If max_chunk is given, return the largest divisor <= max_chunk.
    Otherwise, return roughly half of n (or closest smaller divisor).
    """
    divisors = [i for i in range(2, n) if n % i == 0]  # skip 1
    if not divisors:
        return n  # prime, use full width

    if max_chunk:
        divisors = [d for d in divisors if d <= max_chunk]
        return divisors[-1] if divisors else n

    # Default: pick roughly half
    half = n // 2
    smaller_divisors = [d for d in divisors if d <= half]
    return smaller_divisors[-1] if smaller_divisors else divisors[0]

def display_df_in_chunks(df: pd.DataFrame | pd.io.formats.style.Styler, chunk_size: int = 15) -> None:
    """
    Display a wide DataFrame in fixed-size column chunks, stacked vertically.
    Works with both regular DataFrames and Styler objects.
    """

    # Handle Styler objects
    if isinstance(df, Styler):
        styler = df
        df = styler.data
    else:
        styler = None

    n_cols = df.shape[1]

    for start in range(0, n_cols, chunk_size):
        chunk = df.iloc[:, start:start + chunk_size]
        if styler:
            display(chunk.style)
        else:
            display(chunk)
                
def list_json_files() -> list:
    """
    List all JSON files in the 'json_cache' directory without the '.json' extension.
    
    Returns:
        list: List of JSON file names without extension.
    """
    if not os.path.exists('json_cache'):
        os.makedirs('json_cache')
    return [f for f in os.listdir('json_cache') if f.endswith('.json')]

def load_from_json(file_name: str) -> dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        file_name (str): Path to the JSON file.
        
    Returns:
        dict: Parsed JSON data.
    """
    package_dir = os.path.dirname(os.path.dirname(__file__))
    if not file_name.endswith('.json'):
        file_name += '.json'

    file_path = os.path.join(package_dir, "json_cache", file_name)
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def save_to_json(data: dict[str, Any], file_name: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data (dict): Data to save.
        file_name (str): Path to the JSON file.
    """
    package_dir = os.path.dirname(os.path.dirname(__file__))
    json_cache_dir = os.path.join(package_dir, "json_cache")
    file_name = os.path.join(json_cache_dir, file_name if file_name.endswith('.json') else f"{file_name}.json")
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def delete_json_file(file_name: str) -> None:
    """
    Delete a JSON file if it exists.

    Args:
        file_name (str): Path to the JSON file.
    """
    package_dir = os.path.dirname(os.path.dirname(__file__))
    json_cache_dir = os.path.join(package_dir, "json_cache")
    file_name = os.path.join(json_cache_dir, file_name if file_name.endswith('.json') else f"{file_name}.json")
    if os.path.exists(file_name):
        os.remove(file_name)

def extract_total_amount(text: str) -> int | None:
    text_lower = text.lower()
    text_lower = re.sub(r'[^\x00-\x7F]+', ',', text_lower)
    number_matches = list(re.finditer(r'\b\d{1,3},000\b', text_lower))

    totaal_match = re.search(r'totaal aantal', text_lower)
    if not totaal_match or not number_matches:
        return None

    return int(number_matches[0].group().replace(',', '')[:-3])

def find_intersections(x_flow: list[float], total_flow_rate: list[float], x_spec: list[float], min_flow_rate: list[float],\
                        max_flow_rate: list[float], resolution: int = 100) -> dict:
    """
    Find all (x, y) points where total_flow_rate crosses min_flow_rate or max_flow_rate,
    and compute slopes of all three curves over the refined domain.

    Parameters:
        x_flow (array-like): X-values for total flow rate (green line).
        total_flow_rate (array-like): Y-values for total flow rate.
        x_spec (array-like): X-values for spec lines.
        min_flow_rate (array-like): Y-values for minimum spec line.
        max_flow_rate (array-like): Y-values for maximum spec line.
        resolution (int): Number of points in fine-grained domain.

    Returns:
        dict: {
            'intersections': [(x1, y1), (x2, y2), ...],
            'flow_slope': array of slopes,
            'min_slope': array of slopes,
            'max_slope': array of slopes
        }
    """
    # Create fine-grained common domain
    x_min = max(min(x_flow), min(x_spec))
    x_max = min(max(x_flow), max(x_spec))
    x_fine = np.linspace(x_min, x_max, resolution)

    # Interpolate all curves onto fine domain
    flow_interp = interp1d(x_flow, total_flow_rate, bounds_error=False, fill_value=np.nan)
    min_interp = interp1d(x_spec, min_flow_rate, bounds_error=False, fill_value=np.nan)
    max_interp = interp1d(x_spec, max_flow_rate, bounds_error=False, fill_value=np.nan)

    flow_vals = flow_interp(x_fine)
    min_vals = min_interp(x_fine)
    max_vals = max_interp(x_fine)

    # Compute slopes using finite differences
    dx = np.diff(x_fine)
    flow_slope = np.diff(flow_vals) / dx
    min_slope = np.diff(min_vals) / dx
    max_slope = np.diff(max_vals) / dx

    # Find intersections
    intersections = []

    def detect_crossings(diff_array, y_array):
        for i in range(len(diff_array) - 1):
            if np.isnan(diff_array[i]) or np.isnan(diff_array[i+1]):
                continue
            if diff_array[i] * diff_array[i+1] < 0:
                t = abs(diff_array[i]) / (abs(diff_array[i]) + abs(diff_array[i+1]))
                x_cross = x_fine[i] + t * (x_fine[i+1] - x_fine[i])
                y_cross = y_array[i] + t * (y_array[i+1] - y_array[i])
                intersections.append((x_cross, y_cross))

    detect_crossings(flow_vals - min_vals, flow_vals)
    detect_crossings(flow_vals - max_vals, flow_vals)

    return {
        'intersections': intersections,
        'flow_slope': np.average(flow_slope),
        'min_slope': np.average(min_slope),
        'max_slope': np.average(max_slope)
    }

def get_slope(x: list[float], y: list[float], resolution: int = 100) -> float:
    """
    Calculate the average slope of y with respect to x using finite differences.

    Parameters:
        x (array-like): X-values.
        y (array-like): Y-values.

    Returns:
        float: Average slope (dy/dx).
    """
    x = np.array(x)
    y = np.array(y)

    x_fine = np.linspace(min(x), max(x), resolution)
    y_interp = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
    y_fine = y_interp(x_fine)
    dx = np.diff(x_fine)
    dy = np.diff(y_fine)
    slopes = dy / dx
    return np.mean(slopes)


def plot_distribution(array: list[float] | None = None, part_name: str | None = None, tv_id: int | None = None, value: float | None = None,\
                       nominal: float | None = None, bins: int = 50, title: str = "Distribution", xlabel: str = "Values", ylabel: str = "Frequency") -> None:
    """
    Plots a histogram/distribution of an array and marks a specific value and optional nominal value on it.
    
    Parameters:
        array (array-like): The data to plot.
        value (float): The specific value to indicate on the distribution.
        nominal (float): The nominal/target value to indicate on the distribution.
        bins (int): Number of bins for the histogram (more bins = narrower bars, closer spacing).
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """

    plt.close('all')
    array = np.array(array, dtype=float)

    plt.figure(figsize=(10, 5))
    plt.hist(array, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')

    # Mark the actual value
    if value is not None:
        plt.axvline(value, color='red', linestyle='--', linewidth=2, label=f'{part_name} TV ID {tv_id}: {value} [mm]')

    # Mark the nominal value if provided
    if nominal is not None:
        plt.axvline(nominal, color='green', linestyle='-', linewidth=2, label=f'Nominal = {nominal}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_simulated_distribution(nominal: float, min_val: float, max_val: float, value: float | None = None, part_name: str | None = None, tv_id: int | None = None, \
                                bins: int = 50, title: str = "Simulated Distribution", xlabel: str = "Dimension [mm]", ylabel: str = "Frequency", n_samples: int = 10000) -> None:
    """
    Simulates a normal distribution based on a nominal, min, and max value, and plots it.
    
    Parameters:
        nominal (float): The nominal/target value (mean of the distribution).
        min_val (float): The minimum value (used to estimate std deviation).
        max_val (float): The maximum value (used to estimate std deviation).
        value (float): A specific value to mark on the distribution.
        bins (int): Number of bins for the histogram.
        n_samples (int): Number of samples to generate in the simulated distribution.
        title (str): Plot title.
        xlabel (str): X-axis label.
        ylabel (str): Y-axis label.
    """
    # Estimate standard deviation assuming 99.7% of values lie within min-max (3σ rule)
    sigma = (max_val - min_val) / 6
    np.random.seed(42)
    simulated_data = np.random.normal(loc=nominal, scale=sigma, size=n_samples)

    plt.figure(figsize=(10, 5))
    plt.hist(simulated_data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')

    # Mark the actual value
    if value is not None:
        plt.axvline(value, color='red', linestyle='--', linewidth=2,
                    label=f'{part_name} TV ID {tv_id}: {value}' if part_name and tv_id else f'Value: {value}')

    # Mark the nominal value
    plt.axvline(nominal, color='green', linestyle='-', linewidth=2, label=f'Nominal = {nominal}')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def linear_regression(X: np.ndarray | list, y: np.ndarray | list) -> tuple:
    """
    Perform linear regression on the given data.
    Parameters:
        X (array-like): Independent variable(s).
        y (array-like): Dependent variable.
    Returns:
        model: Trained LinearRegression model.
        y_pred: Predicted values.
        coef: Coefficients of the regression.
        intercept: Intercept of the regression.
    """
    X = np.array(X)
    y = np.array(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    coef = model.coef_
    intercept = model.intercept_
    return model, y_pred, coef, intercept

def show_modal_popup(message: str, continue_action: callable) -> None:
    """
    Display a modal popup that floats above all other widgets.
    continue_action: function executed when 'Continue Anyway' is clicked.
    """
    dialog = v.Dialog(
        v_model=True,
        persistent=True,
        max_width="500px",
        style_="position: fixed; top: 20%; left: 50%; transform: translate(-50%, 0); z-index: 9999;"
    )

    card = v.Card(children=[
        v.CardTitle(children=["Confirmation Required"]),
        v.CardText(children=[message]),
        v.CardActions(children=[
            v.Spacer(),
            v.Btn(children=["Cancel"], color="grey", text=True),
            v.Btn(children=["Continue Anyway"], color="red", text=True)
        ])
    ])

    dialog.children = [card]

    # Button handlers
    def on_cancel(widget, event, data):
        dialog.v_model = False

    def on_continue(widget, event, data):
        dialog.v_model = False
        continue_action()

    card.children[-1].children[1].on_event('click', on_cancel)
    card.children[-1].children[2].on_event('click', on_continue)

    display(dialog)