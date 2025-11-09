# Configuration
config = {
    # Data
    "data_root": "./data",
    "img_size": 128,
    "batch_size": 16,
    # Training
    "num_epochs": 200,
    "gen_lr": 2e-4,
    "disc_lr": 2e-4,
    "lr_decay_epoch": 50,
    # Loss weights
    "lambda_recon": 10.0,
    "lambda_adv": 1.0,
    "lambda_id": 5.0,
    "lambda_gp": 10.0,
    # Training options
    "use_gradient_penalty": True,
    "use_cycle_consistency": True,
    "use_identity_loss": True,
    "gen_update_freq": 1,
    # Saving intervals
    "sample_interval": 5,
    "checkpoint_interval": 10,
    "plot_interval": 10,
    # Directories
    "checkpoint_dir": "./checkpoints",
    "sample_dir": "./samples",
    # Resume training
    "resume_training": False,
}
