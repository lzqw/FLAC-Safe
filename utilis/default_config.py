from utilis.config import Config
# test tau and batchsizie

default_config = Config({
    "seed": 0,
    "tag": "default",
    "start_steps": 5000,
    "cuda": True,
    "num_steps": 1000001,
    "save": True,
    
    "eval": True,

    "eval_numsteps": 10000,
    "eval_times": 5,
    "replay_size": 1000000,

    "algo": "FlowAC",
    "policy": "Flow", 
    "steps": 1,
    "gamma": 0.99, 
    "tau": 0.1,
    "lr": 0.0003, #0.0003
    "batch_size": 256, 
    "updates_per_step": 1,
    "target_update_interval": 2, # for delayed policy update and target network update
    "hidden_size": 512,

    "safe_env": False,
    "cost_gamma": 0.97,
    "safe_threshold": 0.1,
    "safe_bandwidth": 0.05,
    "lambda_safe": 1.0,
    "lambda_jvp": 0.05,
    "jvp_warmup_steps": 20000,
    "jvp_mode": "grad",
    "normalize_jvp": False,
    "jvp_norm_mode": "exact",
    "jvp_hutchinson_samples": 1,
    "jvp_eps": 1e-6,
    "soft_normal_masking": False,
    "masking_warmup_steps": 20000,
    "mask_beta_max": 0.5,
    "mask_beta_tau": 10000,
    "mask_noise_scale": 0.01,
    "mask_noise_clip": 0.25,
    "binary_cost": True,
    "safe_policy_loss": True,

    "target_kinetic_coef": 2.5,
    "init_log_alpha": -2.0,
    "auto_alpha": True,
    "compile_model": False,

    "normalize_obs": True,
    "obs_norm_clip": 10.0,
    "obs_norm_eps": 1e-8,

    "distributional_critic": False,
    "critic_num_atoms": 101,
    "critic_v_min": -10.0,
    "critic_v_max": 150.0,

})
