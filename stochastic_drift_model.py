import numpy as np
from pyabc import (ABCSMC, RV, Distribution, PNormDistance)
from pyabc.sampler import MulticoreEvalParallelSampler

# Data to fit model to
DEN_TIMES = np.array([0, 10, 30, 90, 180, 365, 545])
DEN_MEANS = np.array([2.116, 1.012, 0.296, 0.249, 0.208, 0.081])

# Model functions
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

def drift_model(params):
    lesions_surviving = surv_drift(DEN_TIMES, params['division_rate'],
                                       params['delta'],
                                       params['lesion_starting_cells'])[1:] * params['initial_lesion_density']
    diff = lesions_surviving - DEN_MEANS
    diff_sq = diff**2
    den_distance = diff_sq.sum()
    if np.isnan(den_distance):
        den_distance = np.inf
    return {'distance': den_distance}


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

          ]

param_order = ['division_rate', 'delta', 'lesion_starting_cells', 'initial_lesion_density']
priors = {}
for p, b in zip(param_order, BOUNDS):
    priors[p] = RV("uniform", b[0], b[1]-b[0])

priors = Distribution(priors)

NUM_CORES = 3
POPULATION_SIZE = 10000
DB_PATH = "drift_model_abc.db"

###### ABC ######
distance = PNormDistance(p=1)
sampler = MulticoreEvalParallelSampler(n_procs=NUM_CORES)

abc = ABCSMC(drift_model, priors, distance, population_size=POPULATION_SIZE)

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

