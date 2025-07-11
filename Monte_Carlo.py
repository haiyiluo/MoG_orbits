import numpy as np
import re
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
from OD2_Luo import orbit_elements

#note: Be sure to import OD2 first! Also, refer to the example dataset for format

print('1929RO_input')
def parse_ra_string(ra_string: str) -> tuple[int, int, float]:
    parts = re.split(r"[:\s]+", ra_string.strip())
    h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return h, m, s

#tbl = fits.open("corr.fits")[1].data
#rms_ra = 3600*(np.mean((tbl.field_ra - tbl.index_ra)**2.))**0.5

def parse_dec_string(dec_string: str) -> tuple[int, int, float, bool]:
    dec_string = dec_string.strip()
    is_neg = dec_string[0] == "-"
    parts = re.split(r"[:\s]+", dec_string[1:])  # Skip sign
    d, m, s = int(parts[0]), int(parts[1]), float(parts[2])
    return d, m, s, is_neg

def convert_hms_to_deg(h: int, m: int, s: float) -> float:
    return 15.0 * (h + m / 60 + s / 3600)

def convert_dms_to_deg(d: int, m: int, s: float, is_neg: bool) -> float:
    deg = d + m / 60 + s / 3600
    return -deg if is_neg else deg

def f_g(r: np.array, v: np.array, tau: float) -> tuple[float, float]:
    r_mag = np.linalg.norm(r)
    f = 1 - tau**2/(2*r_mag**3) + np.dot(r, v)*tau**3/(2*r_mag**5) + \
        tau**4/(24*r_mag**3)*(3*(np.dot(v, v)/r_mag**2 - 1/r_mag**3) - 
        15*(np.dot(r, v)/r_mag**2)**2 + 1/r_mag**3)
    g = tau - tau**3/(6*r_mag**3) + np.dot(r, v)*tau**4/(4*r_mag**5)
    return f, g

def rho_hat(ra_s: str, dec_s: str) -> np.array:
    ra_deg = convert_hms_to_deg(*parse_ra_string(ra_s))
    dec_deg = convert_dms_to_deg(*parse_dec_string(dec_s))
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    '''ra_rad = np.deg2rad(*parse_dec_string(dec_s))
    dec_rad = np.deg2rad(*parse_dec_string(dec_s))'''
    return np.array([
        np.cos(ra_rad) * np.cos(dec_rad),
        np.sin(ra_rad) * np.cos(dec_rad),
        np.sin(dec_rad)
    ])

def percent_error(est, true):
        est = np.array(est)
        true = np.array(true)
        return np.abs((est - true) / true) * 100

def compute_orbital_elements(row: list, noise_level: float =0.0198400287273222/3600) -> tuple:
    epsilon = np.deg2rad(-23.5)
    R_obliquity = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon), -np.sin(epsilon)],
        [0, np.sin(epsilon), np.cos(epsilon)]
    ])
    inverse_r_obliquity = np.array([
        [1, 0, 0],
        [0, np.cos(epsilon), np.sin(epsilon)],
        [0, -np.sin(epsilon), np.cos(epsilon)]
    ])

    stripped_row = [element.strip() for element in row]
    
    # Add noise to sun vectors
    def add_noise(vec_str):
        vec = [float(x) for x in vec_str]
        return inverse_r_obliquity @ (np.array(vec) + 
               np.random.normal(0, noise_level, 3)) #mean: 0, standard deviation: noise level, trials: 3

    ra1s, dec1s = stripped_row[2], stripped_row[3]
    r1 = add_noise(stripped_row[4:7])
    
    ra2s, dec2s = stripped_row[8], stripped_row[9]
    r2 = add_noise(stripped_row[10:13])
    
    ra3s, dec3s = stripped_row[14], stripped_row[15]
    r3 = add_noise(stripped_row[16:19])
    
    # JPL reference data
    r_jpl = np.array([float(x) for x in stripped_row[20:23]])
    v_jpl = np.array([float(x) for x in stripped_row[23:26]])
    orbital_jpl = np.array([float(x) for x in stripped_row[27:33]])

    ρ1hat = rho_hat(ra1s, dec1s)
    ρ2hat = rho_hat(ra2s, dec2s)
    ρ3hat = rho_hat(ra3s, dec3s)
    
    R1 = np.array(r1)
    R2 = np.array(r2)
    R3 = np.array(r3)

    t1 = float(stripped_row[1])
    t2 = float(stripped_row[7])
    t3 = float(stripped_row[13])

    k = 2 * np.pi / 365.25
    tau1 = k * (t1 - t2)
    tau0 = k * (t3 - t1)
    tau3 = k * (t3 - t2)

    # Initial values
    a1 = tau3 / tau0
    a3 = -tau1 / tau0
    
    # Compute initial rhos
    cross_ρ1ρ2_ρ3 = np.dot(np.cross(ρ1hat, ρ2hat), ρ3hat)
    rho1 = (a1 * np.dot(np.cross(R1, ρ2hat), ρ3hat) - 
            np.dot(np.cross(R2, ρ2hat), ρ3hat) + 
            a3 * np.dot(np.cross(R3, ρ2hat), ρ3hat)) / (a1 * cross_ρ1ρ2_ρ3)
    
    rho2 = (a1 * np.dot(np.cross(ρ1hat, R1), ρ3hat) - 
            np.dot(np.cross(ρ1hat, R2), ρ3hat) + 
            a3 * np.dot(np.cross(ρ1hat, R3), ρ3hat)) / (-cross_ρ1ρ2_ρ3)
    
    cross_ρ2ρ3_ρ1 = np.dot(np.cross(ρ2hat, ρ3hat), ρ1hat)
    rho3 = (a1 * np.dot(np.cross(ρ2hat, R1), ρ1hat) - 
            np.dot(np.cross(ρ2hat, R2), ρ1hat) + 
            a3 * np.dot(np.cross(ρ2hat, R3), ρ1hat)) / (a3 * cross_ρ2ρ3_ρ1)

    # Initial position vectors
    r1_vec = rho1 * ρ1hat - R1
    r2_vec = rho2 * ρ2hat - R2
    r3_vec = rho3 * ρ3hat - R3

    # Initial velocity estimates
    v12 = (r2_vec - r1_vec) / (t2 - t1)
    v23 = (r3_vec - r2_vec) / (t3 - t2)
    v2 = ((t3 - t2) * v12 + (t2 - t1) * v23) / (t3 - t1)

    # Initial f and g functions
    f1, g1 = f_g(r2_vec, v2, tau1)
    f3, g3 = f_g(r2_vec, v2, tau3)

    a1_recal = g3 / (f1 * g3 - f3 * g1)
    a3_recal = -g1 / (f1 * g3 - f3 * g1)

    # Main iteration loop
    r2_new = np.array([1, 1, 1])
    v2_new = np.array([1, 1, 1])
    iteration_errors = []
    
    while True: 
        # Recompute rhos with updated coefficients
        rho1 = (a1_recal * np.dot(np.cross(R1, ρ2hat), ρ3hat) - 
                np.dot(np.cross(R2, ρ2hat), ρ3hat) + 
                a3_recal * np.dot(np.cross(R3, ρ2hat), ρ3hat)) / (a1_recal * cross_ρ1ρ2_ρ3)
        
        rho2 = (a1_recal * np.dot(np.cross(ρ1hat, R1), ρ3hat) - 
                np.dot(np.cross(ρ1hat, R2), ρ3hat) + 
                a3_recal * np.dot(np.cross(ρ1hat, R3), ρ3hat)) / (-cross_ρ1ρ2_ρ3)
        
        rho3 = (a1_recal * np.dot(np.cross(ρ2hat, R1), ρ1hat) - 
                np.dot(np.cross(ρ2hat, R2), ρ1hat) + 
                a3_recal * np.dot(np.cross(ρ2hat, R3), ρ1hat)) / (a3_recal * cross_ρ2ρ3_ρ1)

        # Update position vectors
        r1_vec = rho1 * ρ1hat - R1
        r2_vec = rho2 * ρ2hat - R2
        r3_vec = rho3 * ρ3hat - R3
        
        
        # Update f and g functions
        f1, g1 = f_g(r2_new, v2_new, tau1)
        f3, g3 = f_g(r2_new, v2_new, tau3)

        # Update position and velocity
        r2_new = (g3 * r1_vec - g1 * r3_vec) / (f1 * g3 - f3 * g1)
        v2_new = (f3 * r1_vec - f1 * r3_vec) / (f3 * g1 - f1 * g3) * k

        # Update coefficients
        a1_recal = g3 / (f1 * g3 - f3 * g1)
        a3_recal = -g1 / (f1 * g3 - f3 * g1)

        # Convert to ecliptic frame
        r2_final = R_obliquity @ r2_new
        v2_final = R_obliquity @ v2_new
        

        # Compute orbital elements
        '''state_vector = np.concatenate((v2_final, r2_final))
        elements_final = np.array(orbit_elements(list(state_vector)))'''
        elements_final=orbit_elements([(R_obliquity@r2_new)[0], (R_obliquity@r2_new)[1],(R_obliquity@r2_new)[2], (R_obliquity@v2_new)[0], (R_obliquity@v2_new)[1], (R_obliquity@v2_new)[2]])
                
        print('final_elements',elements_final)
        
        # Calculate errors
        error_percent = percent_error(elements_final, orbital_jpl)
        iteration_errors.append(error_percent)
        
        # Check convergence
        if np.linalg.norm(r2_new - r2_vec) / np.linalg.norm(r2_new) < 1e-10:
            break

    return elements_final, orbital_jpl, iteration_errors

def compD(fname: str, monte_carlo_runs: int = 100, noise_level: float = 1e-6):
    # Initialize error storage
    all_errors = []
    element_names = ['a', 'e', 'i', 'Ω', 'ω', 'M']
    
    # Read CSV file
    with open(fname, 'r') as file:
        reader = csv.reader(file)
        print('reader:', fname)
        next(reader)  # Skip header
        
        # Process each row (each asteroid)
        for row_idx, row in enumerate(reader):
            print(f"\nProcessing asteroid {row_idx+1}")
            row_errors = {name: [] for name in element_names}
            mc_results = []
            element_values = {name: [] for name in element_names} #creates dictionary name with 'element name'as the name of each list of element

            # Monte Carlo simulation
            for run in tqdm(range(monte_carlo_runs), desc="Monte Carlo runs"):
                elements_final, orbital_jpl, iter_errors = compute_orbital_elements(row, noise_level)
                mc_results.append(elements_final)
                
                # Store final errors
                final_error = percent_error(elements_final, orbital_jpl)
                for i, name in enumerate(element_names):
                    row_errors[name].append(final_error[i])

                element_values = {name: [] for name in element_names}  # NEW LINE

            for run in tqdm(range(monte_carlo_runs), desc="Monte Carlo runs"):
                elements_final, orbital_jpl, iter_errors = compute_orbital_elements(row, noise_level)
                mc_results.append(elements_final)
                
                # Store final values
                for i, name in enumerate(element_names):
                    element_values[name].append(elements_final[i])
                
                # Store final errors
                final_error = percent_error(elements_final, orbital_jpl)
                for i, name in enumerate(element_names):
                    row_errors[name].append(final_error[i])

            # Convert to numpy arrays
            for name in element_names:
                row_errors[name] = np.array(row_errors[name])
            
            # Compute statistics
            means = {name: np.mean(row_errors[name]) for name in element_names}
            stds = {name: np.std(row_errors[name]) for name in element_names}
            
            # Print results
            print("\nOrbital Element Error Statistics:")
            print(f"{'Element':<5} {'Mean Error (%)':<15} {'Std Dev (%)':<15}")
            for name in element_names:
                print(f"{name:<5} {means[name]:<15.6f} {stds[name]:<15.6f}")
            
            # Plotting
            plt.figure(figsize=(15, 10))
            
            # Error distribution
            plt.subplot(2, 2, 1)
            plt.boxplot([row_errors[name] for name in element_names])
            plt.xticks(range(1, len(element_names)+1), element_names)
            plt.title('Orbital Element Error Distribution')
            plt.ylabel('Percent Error')
            plt.grid(True)
            
            # Mean errors
            plt.subplot(2, 2, 2)
            plt.bar(element_names, [means[name] for name in element_names], 
                    yerr=[stds[name] for name in element_names], capsize=5)
            plt.title('Mean Percent Errors with Standard Deviation')
            plt.ylabel('Percent Error')
            plt.grid(True)
         
            
            # Convergence plot — use only the last run's iteration errors for legend clarity
            plt.subplot(2, 1, 2)
            if len(iter_errors) > 0:
                iter_errors = np.array(iter_errors)  # Shape: (num_iterations, num_elements)
                for i, name in enumerate(element_names):
                    if i < iter_errors.shape[1]:
                        plt.plot(iter_errors[:, i], label=name)
            plt.title('Error Convergence During Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Percent Error')
            plt.legend()
            plt.grid(True)


            plt.tight_layout()
            plt.savefig(f'asteroid_{row_idx+1}_error_analysis.png')
            plt.close()
            
            # Histogram of orbital element values
            plt.figure(figsize=(18, 10))
            for i, name in enumerate(element_names):
                plt.subplot(2, 3, i+1)
                plt.hist(element_values[name], bins=20, edgecolor='black')
                plt.title(f'Distribution of {name}')
                plt.xlabel(f'{name} value')
                plt.ylabel('Frequency')
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'asteroid_{row_idx+1}_element_histograms.png')
            plt.close()

            # Store results for final summary
            all_errors.append({
                'means': means,
                'stds': stds,
                'elements_final': elements_final,
                'orbital_jpl': orbital_jpl
            })
    
    # Final summary
    print("\n\n=== FINAL===")
    for idx, result in enumerate(all_errors):
        print(f"\nAsteroid {idx}:")
        print(f"{'Element':<5} {'Computed':<15} {'JPL':<15} {'Error (%)':<10}")
        for i, name in enumerate(element_names):
            computed = result['elements_final'][i]
            jpl_val = result['orbital_jpl'][i]
            error = abs(computed - jpl_val) / jpl_val * 100
            print(f"{name:<5} {computed:<15.6f} {jpl_val:<15.6f} {error:<10.6f}")
    
    return len(all_errors)

# Run the analysis

print(compD("1929RO_input.csv", monte_carlo_runs=100000, noise_level=0.02/3600))
#print(compD("mog_test_cases.csv", monte_carlo_runs=100, noise_level=10e-6))
