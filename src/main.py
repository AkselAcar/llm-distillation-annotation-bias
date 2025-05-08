import argparse
from model import DistilBertDistiller, HateSpeechDataset
from utils import read_yml
import constants

def main(args: argparse.Namespace):
    """Define the main function to run the project.

    Args:
        args (argparse.Namespace) : Arguments from the command line.
    """
    cfg_file = args.cfg
    cfg = read_yml(cfg_file)
    model_cfg = cfg.get("model", {})
    model = DistilBertDistiller(lr=model_cfg.get("learning_rate", constants.LEARNING_RATE))

    datasets_cfg = cfg.get("datasets", {})
    loader_train, loader_val, loader_test = parse_datasets(cfg=datasets_cfg)

    print("Logging model...")
    log_cfg(cfg=model_cfg)
    log_model_summary(model=model)

    print("Training model...")
    fit(self, train_dataset, val_dataset=None, epochs=3, batch_size=16, temperature=2.0, alpha=0.5)
    model.fit(
        loader_train,
        loader_val,
        epochs=model_cfg.get("n_epochs", constants.N_EPOCHS),
        learning_rate=model_cfg.get("learning_rate", constants.LEARNING_RATE),
        criterion_name=model_cfg.get("criterion", constants.CRITERION),
        optimizer_name=cfg.get("optimizer", constants.OPTIMIZER),
    )

    model.predict(loader=loader_train)

    model.create_submission(loader=loader_test, path=os.path.join(constants.SUBMISSION_DIR, str(run_id) + ".csv"))

if __name__ == "__main__":

    # Use argument
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--cfg", default="config.yml", help="Configuration file")
    parser.add_argument("--dataset_path", default=constants.DATA_DIR, help="Dataset path")
    # parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    args.cfg = os.path.join(constants.PROJECT_DIR, args.cfg)

    # print(constants.ROOT_DIR)

    main(args=args)