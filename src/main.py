import argparse
from model import Llama4Baseline, DistilBertBaseline, DistilBertDistiller
from utils import read_yml, plot_metric_over_epochs
import constants
import os
from dataloader import parse_datasets
from datetime import datetime
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main(args: argparse.Namespace):
    """Define the main function to run the project.

    Args:
        args (argparse.Namespace) : Arguments from the command line.
    """

    start_time = datetime.now()
    cfg_file = args.cfg
    cfg = read_yml(cfg_file)
    model_cfg = cfg.get("model", {})

    
    # model = Llama4Baseline()
    # model = getattr(model, model_cfg.get("name", {}))(device=constants.DEVICE)
    datasets_path = args.datasets_path

    datasets_cfg = cfg.get("datasets", {})
    loader_train, loader_val, loader_test = parse_datasets(cfg=datasets_cfg,
                                                           data_path=datasets_path)

    print("Training model with DAgger + DaD...")
    train_loss_list, val_acc_list, val_f1_list = [],[],[]
    if not args.baseline : # Train the model if not baseline
        # Get alpha value from config
        alpha_cfg = model_cfg.get("alpha", {})
        start, stop, step = alpha_cfg.get("start", 0.0), alpha_cfg.get("stop", 1.0), alpha_cfg.get("step", 0.1)
        alpha_values = np.arange(start,stop+step/2,step)

        teacher_tokenizer = AutoTokenizer.from_pretrained(constants.model_path, padding_side="left", local_files_only=True)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            constants.model_path,
            device_map="auto",
            torch_dtype=torch.float16,  # Load model in half-precision
            local_files_only=True
        )
        for alpha in alpha_values :
            model = DistilBertDistiller(device=constants.DEVICE)
            n_outer_iterations = model_cfg.get("dagger_outer_loops", 3)
            epochs_per_step = model_cfg.get("n_epochs", constants.N_EPOCHS)

            for iteration in range(n_outer_iterations):
                print(f"=== Outer Step {iteration + 1}/{n_outer_iterations} ===")
                model.fit(
                    loader_train,
                    loader_val,
                    epochs=epochs_per_step,
                    lr=model_cfg.get("learning_rate", constants.LEARNING_RATE),
                    batch_size=model_cfg.get("batch_size", constants.BATCH_SIZE),
                    temperature=model_cfg.get("temperature", constants.TEMPERATURE),
                    alpha=alpha,
                    optimizer=model_cfg.get("optimizer", constants.OPTIMIZER),
                    criterion=model_cfg.get("criterion", constants.CRITERION)
                )
                model.learn_from_student_errors(
                    train_loader=loader_train, teacher_model=teacher_model, n_variants=2
                )

                acc, f1 = model.evaluate(loader=loader_val)
                val_acc_list.append(acc)
                val_f1_list.append(f1)

    plot_metric_over_epochs(val_acc_list, "Validation Accuracy", alpha_values, 
                constants.OUTPUTS_DIR / str("val_accuracy_" + start_time.strftime('%d_%m_%y_%H_%M') +".png"))
    plot_metric_over_epochs(val_f1_list, "Validation F1 Score", alpha_values, 
                constants.OUTPUTS_DIR / str("val_f1_" + start_time.strftime('%d_%m_%y_%H_%M') +".png"))    
    
    # Evaluate on test data and save the results    
    acc, f1 = model.evaluate(loader=loader_test)    

    # Put the configuration, results and run_id in a .csv file
    end_time = datetime.now()
    out_dict = {'Start' : start_time.strftime('%d/%m/%y %H:%M:%S'),
                'End' : end_time.strftime('%d/%m/%y %H:%M:%S'),
                'Accuracy' : acc,
                'F1' : f1}
    # Add model config
    out_dict.update(model_cfg)
    # Add dataset config
    out_dict.update({dataset['set'] + "[split]": dataset['name'] + str(dataset['split']) for dataset in datasets_cfg})
    print(out_dict)
    # print("DATASET CONFIG", datasets_cfg)
    out_df = pd.DataFrame([out_dict])
    file_path = constants.OUTPUTS_DIR / "logs.csv"
    file_exists = os.path.isfile(file_path)
    out_df.to_csv(file_path, index=False, mode='a', header=not file_exists)

if __name__ == "__main__":

    # Use argument
    parser = argparse.ArgumentParser(description="File creation script.")
    parser.add_argument("--cfg", default="config.yml", help="Configuration file")
    parser.add_argument("--datasets_path", default=constants.DATA_DIR, help="Datasets path")
    parser.add_argument("--baseline", default=False, help="Compute DistilBERT baseline")
    # parser.add_argument("--results_path", required=True, help="Output directory")

    args = parser.parse_args()

    args.cfg = os.path.join(constants.ROOT_DIR, args.cfg)

    main(args=args)