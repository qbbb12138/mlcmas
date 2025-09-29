# -*- coding: utf-8 -*-
import os
import re
import math
import pandas as pd
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
zero_file_path = os.path.join(current_dir, 'zero.csv')
phase_file_path = os.path.join(current_dir, 'rare_earth_combinations.csv')
output_csv_path = os.path.join(current_dir, 'calculated_properties.csv')

SUBSCRIPT_MAP = str.maketrans({
    u'₀': '0', u'₁': '1', u'₂': '2', u'₃': '3', u'₄': '4',
    u'₅': '5', u'₆': '6', u'₇': '7', u'₈': '8', u'₉': '9'
})

def normalize_formula(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace('\t', ' ')
    s = re.sub(r'\s+', '', s)          
    s = s.translate(SUBSCRIPT_MAP)      
    return s

zero_data = pd.read_csv(zero_file_path).dropna(how='all').dropna()
zero_data.columns = [c.strip() for c in zero_data.columns]

required_cols = {
    'element', 'r', 'electronegativities', 'energy', 'enthalpy',
    'lambda', 'Tm', 'K'
}
missing = required_cols - set(zero_data.columns)
if missing:
    raise KeyError("Columns missing in zero.csv: {}".format(", ".join(sorted(missing))))

zero_data['element'] = zero_data['element'].astype(str).str.strip()

lookup = {}
for col in ['r', 'electronegativities', 'energy', 'enthalpy', 'lambda', 'Tm', 'K']:
    lookup[col] = dict(zip(zero_data['element'], zero_data[col]))

phase_data = pd.read_csv(phase_file_path).dropna(how='all').dropna()
phase_data.columns = [c.strip() for c in phase_data.columns]
if 'Compounds' not in phase_data.columns:
    raise KeyError("Column 'Compounds' not found in rare_earth_combinations.csv")

phase_data['Compounds'] = phase_data['Compounds'].astype(str)
phase_data['Compounds_norm'] = phase_data['Compounds'].apply(normalize_formula)

mask_SiO5 = phase_data['Compounds_norm'].str.contains(r'SiO5\b', regex=True, na=False)
mask_Si1O5 = phase_data['Compounds_norm'].str.contains(r'Si1O5\b', regex=True, na=False)
mask_remove = mask_SiO5 | mask_Si1O5
after_drop = phase_data[~mask_remove].copy()

has_Si2O7_any = after_drop['Compounds_norm'].str.contains(r'Si2O7\b', regex=True, na=False).any()
if has_Si2O7_any:
    after_drop = after_drop[after_drop['Compounds_norm'].str.contains(r'Si2O7\b', regex=True, na=False)].copy()
    fallback_used = False
else:
    fallback_used = True
    print("[Warning] No 'Si2O7' rows found after removing 'SiO5'. "
          "Proceeding with all remaining rows (SiO5-removed).")

if after_drop.empty:
    raise ValueError("No rows remain after filtering. Check 'Compounds' spellings/spaces/subscripts.")

def strip_silicate_tail(formula: str) -> str:
    return re.sub(r'(Si\d*O\d*)$', '', normalize_formula(formula))

after_drop['Compounds_clean'] = after_drop['Compounds'].apply(strip_silicate_tail)

def parse_composition(comp_str: str):
    s = normalize_formula(comp_str) 
    pattern = r'([A-Z][a-z]?)([\d\.]*)'
    matches = re.findall(pattern, s)
    comp = {}
    for element, frac in matches:
        if element in ('Si', 'O'):
            continue
        value = 1.0 if frac == '' else float(frac)
        comp[element] = comp.get(element, 0.0) + value
    return comp

after_drop['parsed_components'] = after_drop['Compounds_clean'].apply(parse_composition)

def safe_float_from_cell(x):
    s = str(x).replace('−', '-').strip()
    return float(s)

def avg_by_fraction(comp: dict, col: str) -> float:
    total = 0.0
    s = sum(comp.values())
    if s == 0:
        return 0.0
    lut = lookup[col]
    for el, f in comp.items():
        if el not in lut:
            continue
        val = safe_float_from_cell(lut[el])
        total += (f / s) * val
    return total

def stddev_by_fraction(comp: dict, col: str, mean_val: float) -> float:
    if mean_val == 0:
        return 0.0
    dev = 0.0
    s = sum(comp.values())
    if s == 0:
        return 0.0
    lut = lookup[col]
    for el, f in comp.items():
        if el not in lut:
            continue
        val = safe_float_from_cell(lut[el])
        dev += (f / s) * ((val - mean_val) ** 2)
    return dev ** 0.5

def calculate_configuration_entropy(comp: dict) -> float:
    R_const = 8.314
    s = sum(comp.values())
    if s == 0:
        return 0.0
    acc = 0.0
    for _, f in comp.items():
        if f <= 0:
            continue
        x = f / s
        acc += x * math.log(x)
    return -R_const * acc

def calculate_energy_like(comp: dict, col: str) -> float:
    total = 0.0
    s = sum(comp.values())
    if s == 0:
        return 0.0
    lut = lookup[col]
    for el, f in comp.items():
        if el not in lut:
            continue
        val = safe_float_from_cell(lut[el])
        total += (f / s) * val
    return total

def calculate_Tm_K(comp: dict) -> float:
    avg_Tm_C = avg_by_fraction(comp, 'Tm')
    return avg_Tm_C + 273.15

def calculate_K_from_K(comp: dict) -> float:
    return avg_by_fraction(comp, 'K')

def calculate_properties(comp: dict) -> dict:
    R_avg   = avg_by_fraction(comp, 'r')
    chi_avg = avg_by_fraction(comp, 'electronegativities')
    OB_avg  = avg_by_fraction(comp, 'lambda')
    Tm_K    = calculate_Tm_K(comp)
    K_val   = calculate_K_from_K(comp)

    return {
        'R':         round(R_avg, 5),
        'sigma_R':   round(stddev_by_fraction(comp, 'r', R_avg), 5),
        'chi':       round(chi_avg, 5),
        'sigma_chi': round(stddev_by_fraction(comp, 'electronegativities', chi_avg), 5),
        'OB':        round(OB_avg, 5),
        'sigma_OB':  round(stddev_by_fraction(comp, 'lambda', OB_avg), 5),
        'delta_S':   round(calculate_configuration_entropy(comp), 5),
        'delta_E':   round(calculate_energy_like(comp, 'energy'), 5),
        'delta_H':   round(calculate_energy_like(comp, 'enthalpy'), 5),
        'Tm':        round(Tm_K, 2),
        'K':         round(K_val, 2),
    }

props_df = pd.DataFrame(after_drop['parsed_components'].apply(calculate_properties).tolist())

final_cols = ['R', 'sigma_R', 'chi', 'sigma_chi', 'OB', 'sigma_OB',
              'delta_S', 'delta_E', 'delta_H', 'Tm', 'K']
props_df = props_df[final_cols]

def clean_for_display(s: str) -> str:
    s = str(s).replace('\t', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    return s

after_drop['Compounds'] = after_drop['Compounds'].apply(clean_for_display)

final_data = pd.concat(
    [after_drop[['Compounds']].reset_index(drop=True),
     props_df.reset_index(drop=True)],
    axis=1
)

final_data.to_csv(
    output_csv_path,
    index=False,
    sep=',',
    encoding='utf-8-sig',
    quoting=csv.QUOTE_MINIMAL
)

print("Saved ->", output_csv_path)
if fallback_used:
    print("[Note] Output includes all rows after removing SiO5/Si1O5 (no Si2O7 found).")
