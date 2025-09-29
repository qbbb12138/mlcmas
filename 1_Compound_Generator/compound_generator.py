##############################################################################################
##############################################################################################
###########                                                                        ###########
###########                           Compound Generator                           ###########
###########                                                                        ###########
###########     1.Element Pool: e.g., 'Yb', 'Tm', 'Er', 'Y', 'Ho', 'Tb', 'Gd'      ###########
###########     2.Step size: e.g., 0.05                                            ###########
###########     3.Component number: e.g., 5                                        ###########
###########                                                                        ###########
##############################################################################################
##############################################################################################

import itertools
import pandas as pd
from multiprocessing import Pool, cpu_count

elements = [] #####   1.Element Pool: e.g., 'Yb', 'Tm', 'Er', 'Y', 'Ho', 'Tb', 'Gd'
target_sum = 2.0
step =  #####   2.Step size: e.g., 0.05
phases = ['Si2O7', 'Si1O5']
int_target = int(target_sum / step)
int_step_values = list(range(1, int_target + 1))

def process_combo(combo):
    local_results = []
    r = len(combo)
    for coeffs in itertools.product(int_step_values, repeat=r):
        if sum(coeffs) == int_target and all(c > 0 for c in coeffs):
            formula_part = ''.join(f'{el}{round(c * step, 2)}' for el, c in zip(combo, coeffs))
            for phase in phases:
                local_results.append(formula_part + phase)
    return local_results

def main():
    all_combos = list(itertools.combinations(elements, )) #####   3.Component number: e.g., 5

    print(f'A total of {len(all_combos)} element combinations have been generated. Starting parallel computation...')
    with Pool(processes=cpu_count()) as pool:
        all_results = pool.map(process_combo, all_combos)

    flattened = [item for sublist in all_results for item in sublist]
    df = pd.DataFrame(flattened, columns=['Compounds'])
    df.to_csv('rare_earth_combinations.csv', index=False)
    print('Successfully saved: rare_earth_combinations.csv')

if __name__ == '__main__':
    main()
