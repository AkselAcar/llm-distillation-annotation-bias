import argparse

def main(args: argparse.Namespace):
    """Define the main function to run the project.

    Args:
        args (argparse.Namespace) : Arguments from the command line.
    """
    # Check if config file exists
    cfg_file = args.cfg
    seed = args.seed

    set_seed(seed=seed)
    cfg = read_yml(cfg_file=cfg_file)

    # Get main infos
    experiment = cfg.get("experiment", "Default")
    name = cfg.get("name", "debug")

    # Set mlflow informations
    mlflow.set_tracking_uri(uri="./project/mlruns")
    mlflow.set_experiment(experiment)
    run_name = "{}:{}:{}".format(experiment, name, seed)

    with mlflow.start_run(run_name=run_name):

        run_id = mlflow.active_run().info.run_id

        model_cfg = cfg.get("model", {})
        model = choose_model(cfg=model_cfg)

        datasets_cfg = cfg.get("datasets", {})
        loader_train, loader_val, loader_test = parse_datasets(cfg=datasets_cfg)

        print("Logging model...")
        log_cfg(cfg=model_cfg)
        log_model_summary(model=model)

        print("Training model...")
        model.fit(
            loader_train,
            loader_val,
            num_epochs=model_cfg.get("n_epochs", constants.NUM_EPOCHS),
            learning_rate=model_cfg.get("learning_rate", constants.LEARNING_RATE),
            criterion_name=model_cfg.get("criterion", constants.CRITERION),
            optimizer_name=cfg.get("optimizer", constants.OPTIMIZER),
        )

        model.predict(loader=loader_train)

        model.create_submission(loader=loader_test, path=os.path.join(constants.SUBMISSION_DIR, str(run_id) + ".csv"))

if __name__ == "__main__":

    # Use argument
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--dataset_path", required=True, help="Dataset path")
    parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    main(args=args)