import numpy as np
from pyabc import (ABCSMC, RV, Distribution, PNormDistance)
from pyabc.sampler import MulticoreEvalParallelSampler

# Data to fit model to
DEN_TIMES = np.array([0, 10, 30, 90, 180, 365, 545])
DEN_MEANS = np.array([2.116, 1.012, 0.296, 0.249, 0.208, 0.081])
DBZ_DATA = np.array([0.952, 1.347])

# Model functions
def logistic_clone_growth(times, x0, k):
    return 1/(1+(1-x0)/x0*np.exp(-k*times))

def surv_clone_sweep(clone_sweep):
    initial_clone_proportion = clone_sweep[0]
    remaining_proportion_of_tissue = 1-initial_clone_proportion
    return 1 - (clone_sweep-initial_clone_proportion)/remaining_proportion_of_tissue

# Drift survival probabilities. From Bailey - the elements of stochastic processes
def surv_drift_balanced(t, division_rate, starting_cells):
    # Special case where delta=0
    a = (division_rate*t)/(division_rate*t+2)
    return 1 - a**starting_cells

def surv_drift(t, division_rate, delta, starting_cells):
    if delta == 0:
        # Use the balanced equation
        return surv_drift_balanced(t, division_rate, starting_cells)
    else:
        a = 2*division_rate*delta*t
        b = np.exp(a)
        # For some extreme values, b can be infinite. Replace with max possible floating point value to make sure no nan results.
        np.nan_to_num(b, copy=False)
        c = (1/2-delta) * (b-1)
        d = (1/2+delta) * b - (1/2 - delta)
        alpha = c/d
        return 1-alpha**starting_cells

def surv_combined(t, division_rate, delta, starting_cells, clone_sweep):
    # Combine the drift and clonal competition lesion loss for overall survival probability
    p_surv_drift = surv_drift(t, division_rate, delta, starting_cells)
    p_surv_clone_sweep = surv_clone_sweep(clone_sweep)
    return p_surv_drift*p_surv_clone_sweep

def run_model(params):
    # Parameters determining drift
    division_rate = params['division_rate']
    delta = params['delta']
    lesion_starting_cells = params['lesion_starting_cells']
    
    # Parameters determining resistance/sensitivity
    initial_sensitive_lesion_density = params['sensitive_lesion_proportion']*params['initial_lesion_density']
    initial_resistant_lesion_density = (1-params['sensitive_lesion_proportion'])*params['initial_lesion_density']
    
    # Parameters determining clonal spread in DEN
    starting_mutant_proportion = params['starting_mutant_proportion']
    k = params['k']
    
    # DEN lesion density from model
    clone_sweep = logistic_clone_growth(DEN_TIMES, starting_mutant_proportion, k)
    
    sensitive_lesions = surv_combined(DEN_TIMES, division_rate, delta,
                                      lesion_starting_cells, clone_sweep)[1:] * initial_sensitive_lesion_density
        
    resistant_lesions = surv_drift(DEN_TIMES, division_rate, delta, lesion_starting_cells)[1:] * initial_resistant_lesion_density
                                      
    total_lesions = sensitive_lesions + resistant_lesions
                                      
    # Get the distance between the model values and the data
    diff = total_lesions - DEN_MEANS
    diff_sq = diff**2
    den_distance = diff_sq.sum()
      
    # DBZ experiment fitting
    clone_sweep_ctl = logistic_clone_growth(np.array([0, 24]), starting_mutant_proportion, k)  # Just need the start and end of the clone growth
    clone_sweep_dbz = logistic_clone_growth(np.array([0, 10]), starting_mutant_proportion, k)  # Just need to go up to 10 days when DBZ starts, no growth after than.
      
    ctl_sensitive_lesions = surv_combined(np.array([0, 24]), division_rate, delta,
                                            lesion_starting_cells, clone_sweep_ctl) * initial_sensitive_lesion_density
    dbz_sensitive_lesions = surv_combined(np.array([0, 24]), division_rate, delta,
                                            lesion_starting_cells, clone_sweep_dbz) * initial_sensitive_lesion_density
      
    resistant_lesions = surv_drift(np.array([0, 24]), division_rate, delta, lesion_starting_cells)[1:] * initial_resistant_lesion_density
      
    dbz_model_values = np.array([ctl_sensitive_lesions[1], dbz_sensitive_lesions[1]]) + resistant_lesions  # ctl and dbz mean lesion density
    
    # Get the distance between the model values and the data
    diff = dbz_model_values - DBZ_DATA
    diff_sq = diff**2
    dbz_distance = diff_sq.sum()
                                      
    return {'distance': den_distance + dbz_distance}


## Set up the prior distributions for ABC
BOUNDS = [
          # Symmetric division rate. Division rate around 2.5 per week in normal tissue, similar in Lesions.
          # Symmetric division rate must be smaller.
          (0, 2.5/7),
          
          # Delta. Must be between -0.5 and 0.5.
          (-0.5, 0.5),
          
          # Number of proliferating cells per lesion at time 0.
          (1, 100),
          
          # Density of lesions at time 0.
          # At 10 days, there are approximately 2 lesions per mm2. Allow it to be order of magnitude higher at time 0
          (0, 20),
          
          # Proportion of senstive lesions at time 0
          (0, 1),
          
          # Initial proportion of tissue with lesion-removing clones.
          (0, 1),
          
          # growth rate of clones. Limited by the growth rate of MAML clones against WT tissue (Alcolea et al 2014).
          (0, 0.04)
          ]

param_order = ['division_rate', 'delta', 'lesion_starting_cells', 'initial_lesion_density',
               'sensitive_lesion_proportion', 'starting_mutant_proportion', 'k']
priors = {}
for p, b in zip(param_order, BOUNDS):
    priors[p] = RV("uniform", b[0], b[1]-b[0])

priors = Distribution(priors)

NUM_CORES = 3
POPULATION_SIZE = 10000
DB_PATH = "full_model_abc.db"

###### ABC ######
distance = PNormDistance(p=1)
sampler = MulticoreEvalParallelSampler(n_procs=NUM_CORES)

abc = ABCSMC(run_model, priors, distance, population_size=POPULATION_SIZE)

db_path = ("sqlite:///" + DB_PATH)

abc.new(db_path, {'distance': 0})
history = abc.run(minimum_epsilon=0.001, max_nr_populations=25)

# The database is stored and can be reloaded if needed - see pyabc documentation
# Here just print the median and 95% credible intervals.
import pyabc.visualization.credible as credible

def get_estimate_and_CI_for_param(param, df, w, confidence=0.95):
    vals = np.array(df[param])
    lb, ub = credible.compute_credible_interval(vals, w, confidence)
    median = credible.compute_quantile(vals, w, 0.5)
    return {'median': median, 'CI_lower_bound': lb, 'CI_upper_bound':ub}

df, w = history.get_distribution()  # Get the accepted parameters from the last generation

for p in param_order:
    print(p, get_estimate_and_CI_for_param(p, df, w))
