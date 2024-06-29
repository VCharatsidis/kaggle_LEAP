import numpy as np

num_features = 556
num_targets = 368
seq_length = 60


total_rows = 10091520

# Identify vertically resolved variables and scalar variables
# Assuming variable names are in the format "variable_0", "variable_1", ..., "variable_59" for vertically resolved variables
# and "scalar_variable" for scalar variables

# This is a list of your 25 input variables, you need to populate this with your actual variable names
seq_variables_x = ['state_t', 'state_q0001', 'state_q0002',
                     'state_q0003', 'state_u', 'state_v',
                     'pbuf_ozone', 'pbuf_CH4', 'pbuf_N2O']  # Example vertically resolved variables

scalar_variables_x = ['state_ps', 'pbuf_SOLIN', 'pbuf_LHFLX', 'pbuf_SHFLX',
                    'pbuf_TAUX', 'pbuf_TAUY', 'pbuf_COSZRS', 'cam_in_ALDIF', 'cam_in_ALDIR',
                    'cam_in_ASDIF', 'cam_in_ASDIR', 'cam_in_LWUP', 'cam_in_ICEFRAC', 'cam_in_LANDFRAC',
                    'cam_in_OCNFRAC', 'cam_in_SNOWHLAND']  # Example scalar variables

seq_variables_y = ['ptend_t', 'ptend_q0001', 'ptend_q0002', 'ptend_q0003', 'ptend_u', 'ptend_v']
scalar_variables_y = ['cam_out_NETSW', 'cam_out_FLWDS', 'cam_out_PRECSC', 'cam_out_PRECC', 'cam_out_SOLS',
                      'cam_out_SOLL', 'cam_out_SOLSD', 'cam_out_SOLLD']


REPLACE_FROM = ['ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_20', 'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24', 'ptend_q0002_25', 'ptend_q0002_26']
REPLACE_TO = ['state_q0002_0', 'state_q0002_1', 'state_q0002_2', 'state_q0002_3', 'state_q0002_4', 'state_q0002_5', 'state_q0002_6', 'state_q0002_7', 'state_q0002_8', 'state_q0002_9', 'state_q0002_10', 'state_q0002_11', 'state_q0002_12', 'state_q0002_13', 'state_q0002_14', 'state_q0002_15', 'state_q0002_16', 'state_q0002_17', 'state_q0002_18', 'state_q0002_19', 'state_q0002_20', 'state_q0002_21', 'state_q0002_22', 'state_q0002_23', 'state_q0002_24', 'state_q0002_25', 'state_q0002_26']


OFFICIAL_TARGET_WEIGHTS = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
OFFICIAL_TARGET_WEIGHTS = np.array(OFFICIAL_TARGET_WEIGHTS)

TARGET_WEIGHTS =          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ptend_u = list(range(257, 288))
# TARGET_WEIGHTS = [1 if i in ptend_u else 0 for i in range(len(TARGET_WEIGHTS))]
TARGET_WEIGHTS = np.array(TARGET_WEIGHTS)
# max_weight = TARGET_WEIGHTS.mean()
# TARGET_WEIGHTS_root = TARGET_WEIGHTS / max_weight
# min_target = np.ones(TARGET_WEIGHTS.shape)
#
# Targets_norm = np.log(TARGET_WEIGHTS + min_target)
#
# print(Targets_norm)

# shift = TARGET_WEIGHTS.min() + 1
# TARGET_WEIGHTS_root = (TARGET_WEIGHTS + shift) ** (1/9)
# mean_weights = TARGET_WEIGHTS.mean()
# std_weights = TARGET_WEIGHTS.std()


input_variable_order = [
    'pbuf_SOLIN',      # Incoming solar radiation (primary external driver)
    'pbuf_COSZRS',     # Cosine of solar zenith angle, primarily driven by time of day and geographical location
    'cam_in_ICEFRAC',  # Ice fraction, which is a surface characteristic but relatively independent in this context
    'cam_in_LANDFRAC', # Land fraction, a basic geographical descriptor
    'cam_in_OCNFRAC',  # Ocean fraction, another basic geographical descriptor
    'state_ps',        # Surface pressure, influenced by overlying atmospheric conditions but also a fundamental atmospheric measure
    'state_t',         # Temperature at various levels, affected by radiation, surface interactions, and atmospheric dynamics
    'state_q0001',     # Specific humidity, influenced by temperature and surface properties
    'state_q0002',     # Specific humidity, similar dependencies as state_q0001
    'state_q0003',     # Specific humidity, similar dependencies as state_q0001 and state_q0002
    'state_u',         # Eastward component of the wind velocity, influenced by temperature gradients and surface stress
    'state_v',         # Northward component of the wind velocity, similar influences as state_u
    'pbuf_ozone',      # Ozone concentration, dependent on UV radiation and atmospheric chemistry
    'pbuf_CH4',        # Methane concentration, influenced by surface and biological processes
    'pbuf_N2O',        # Nitrous oxide concentration, similarly influenced by surface and biological activities
    'cam_in_ALDIF',    # Albedo for diffuse light, influenced by surface properties which in turn depend on land, ice, and ocean fractions
    'cam_in_ALDIR',    # Albedo for direct light, similar dependencies as cam_in_ALDIF
    'cam_in_ASDIF',    # Albedo for scattered diffuse light, dependent on albedo for direct and diffuse light
    'cam_in_ASDIR',    # Albedo for scattered direct light, dependent on other albedo measures
    'pbuf_LHFLX',      # Latent heat flux, directly influenced by temperature, humidity, and surface properties
    'pbuf_SHFLX',      # Sensible heat flux, similarly influenced by temperature and surface properties
    'pbuf_TAUX',       # Eastward wind stress, dependent on wind patterns and surface conditions
    'pbuf_TAUY',       # Northward wind stress, dependent on wind patterns and surface conditions
    'cam_in_LWUP',     # Upward longwave radiation, dependent on surface temperature and atmospheric composition
    'cam_in_SNOWHLAND' # Snow height over land, dependent on temperature, precipitation, and land cover characteristics
]

# Map the desired order of variables to their indices in the original tensor
variable_order_indices = [
    seq_variables_x.index(var) if var in seq_variables_x else len(seq_variables_x) + scalar_variables_x.index(var)
    for var in input_variable_order]


all_input_vars = seq_variables_x + scalar_variables_x

print(len(all_input_vars))
for x in all_input_vars:
    if x not in input_variable_order:
        print(x)
