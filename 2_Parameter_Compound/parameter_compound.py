##############################################################################################
##############################################################################################
###########                                                                        ###########
###########                         Parameter Calculation                          ###########
###########                                                                        ###########
##############################################################################################
##############################################################################################
import pandas as pd
import re
import os
import math
from multiprocessing import Pool, cpu_count

current_dir = os.path.dirname(os.path.abspath(__file__))

parameters_file_path = os.path.join(current_dir, 'rare_earth_parameters.csv')
parameters_data = pd.read_csv(parameters_file_path).dropna()

csv_file_path = os.path.join(current_dir, 'dataset_cmas.csv')  #####  dataset_cmas.csv; rare_earth_combinations.csv
phase_data = pd.read_csv(csv_file_path).dropna()

def parse_composition(composition_str):
    pattern = r'([A-Za-z]+)([\d\.]*)'
    matches = re.findall(pattern, composition_str)

    composition_dict = {}
    for element, fraction in matches:
        if fraction == '':
            fraction = 1.0
        else:
            fraction = float(fraction)
        composition_dict[element] = fraction

    return composition_dict

phase_data['parsed_components'] = phase_data['Compounds'].apply(parse_composition)

def calculate_weighted_average(composition_dict, properties_data, property_column):
    total_weighted_value = 0
    total_fraction = sum(composition_dict.values())

    if total_fraction == 0:
        return 0

    for element, fraction in composition_dict.items():
        element_data = properties_data[properties_data['element'] == element]
        if element_data.empty:
            continue
        property_value = float(element_data[property_column].values[0])
        normalized_fraction = fraction / total_fraction
        total_weighted_value += normalized_fraction * property_value

    return total_weighted_value

def calculate_deviation(composition_dict, properties_data, property_column, average_value):
    if average_value == 0:
        return 0 

    deviation = 0
    total_fraction = sum(composition_dict.values())

    if total_fraction == 0:
        return 0 

    for element, fraction in composition_dict.items():
        element_data = properties_data[properties_data['element'] == element]
        if element_data.empty:
            continue
        property_value = float(element_data[property_column].values[0])
        normalized_fraction = fraction / total_fraction
        deviation += normalized_fraction * ((property_value - average_value) ** 2)

    return deviation ** 0.5

def calculate_configuration_entropy(composition_dict):
    R = 8.314
    total_entropy = 0
    total_fraction = sum(composition_dict.values())

    if total_fraction == 0:
        return 0

    for element, fraction in composition_dict.items():
        if fraction == 0:
            continue
        normalized_fraction = fraction / total_fraction
        total_entropy += normalized_fraction * math.log(normalized_fraction)

    return -R * total_entropy

def calculate_properties(composition_dict, properties_data):
    average_r = calculate_weighted_average(composition_dict, properties_data, 'r')
    average_mass = calculate_weighted_average(composition_dict, properties_data, 'mass')
    average_specific = calculate_weighted_average(composition_dict, properties_data, 'specific_heat')
    average_vaporization = calculate_weighted_average(composition_dict, properties_data, 'vaporization')
    average_fusion = calculate_weighted_average(composition_dict, properties_data, 'fusion')
    average_electronegativity = calculate_weighted_average(composition_dict, properties_data, 'electronegativity')
    average_density_re = calculate_weighted_average(composition_dict, properties_data, 'density_re')
    average_modulus_re = calculate_weighted_average(composition_dict, properties_data, 'modulus_re')
    average_Tm_re = calculate_weighted_average(composition_dict, properties_data, 'Tm_re')
    average_energy = calculate_weighted_average(composition_dict, properties_data, 'energy')
    average_formation = calculate_weighted_average(composition_dict, properties_data, 'formation')
    average_enthalpy = calculate_weighted_average(composition_dict, properties_data, 'enthalpy')
    average_lambda = calculate_weighted_average(composition_dict, properties_data, 'lambda')
    average_modulus_reo = calculate_weighted_average(composition_dict, properties_data, 'modulus_reo')
    average_Tm_reo = calculate_weighted_average(composition_dict, properties_data, 'Tm_reo')
    average_density_reo = calculate_weighted_average(composition_dict, properties_data, 'density_reo')
    average_CFS = calculate_weighted_average(composition_dict, properties_data, 'CFS')

    return {
        'delta_S': calculate_configuration_entropy(composition_dict),
        'r': average_r,
        'delta_r': calculate_deviation(composition_dict, properties_data, 'r', average_r),
        'mass': average_mass,
        'delta_mass': calculate_deviation(composition_dict, properties_data, 'mass', average_mass),
        'specific': average_specific,
        'delta_specific': calculate_deviation(composition_dict, properties_data, 'specific_heat', average_specific),
        'vaporization': average_vaporization,
        'delta_vaporization': calculate_deviation(composition_dict, properties_data, 'vaporization', average_vaporization),
        'fusion': average_fusion,
        'delta_fusion': calculate_deviation(composition_dict, properties_data, 'fusion', average_fusion),
        'electronegativity': average_electronegativity,
        'delta_electronegativity': calculate_deviation(composition_dict, properties_data, 'electronegativity', average_electronegativity),
        'density_re': average_density_re,
        'delta_density_re': calculate_deviation(composition_dict, properties_data, 'density_re', average_density_re),
        'modulus_re': average_modulus_re,
        'delta_modulus_re': calculate_deviation(composition_dict, properties_data, 'modulus_re', average_modulus_re),
        'Tm_re': average_Tm_re,
        'delta_Tm_re': calculate_deviation(composition_dict, properties_data, 'Tm_re', average_Tm_re),
        'energy': average_energy,
        'delta_energy': calculate_deviation(composition_dict, properties_data, 'energy', average_energy),
        'formation': average_formation,
        'delta_formation': calculate_deviation(composition_dict, properties_data, 'formation', average_formation),
        'enthalpy': average_enthalpy,
        'delta_enthalpy': calculate_deviation(composition_dict, properties_data, 'enthalpy', average_enthalpy),
        'lambda': average_lambda,
        'delta_lambda': calculate_deviation(composition_dict, properties_data, 'lambda', average_lambda),
        'modulus_reo': average_modulus_reo,
        'delta_modulus_reo': calculate_deviation(composition_dict, properties_data, 'modulus_reo', average_modulus_reo),
        'Tm_reo': average_Tm_reo,
        'delta_Tm_reo': calculate_deviation(composition_dict, properties_data, 'Tm_reo', average_Tm_reo),
        'density_reo': average_density_reo,
        'delta_density_reo': calculate_deviation(composition_dict, properties_data, 'density_reo', average_density_reo),
        'CFS': average_CFS,
        'delta_CFS': calculate_deviation(composition_dict, properties_data, 'CFS', average_CFS)        
    }

if __name__ == '__main__':
    with Pool(cpu_count()) as pool:
        properties_list = pool.starmap(calculate_properties, [(comp, parameters_data) for comp in phase_data['parsed_components']])
    properties_df = pd.DataFrame(properties_list).round(5)

    final_data = pd.concat([phase_data.drop(columns=['parsed_components']), properties_df], axis=1)
    output_file_path = os.path.join(current_dir, 'parameter_compound.csv')  #####  property_cmas.csv; parameter_compound
    final_data.to_csv(output_file_path, index=False, float_format='%.5f')