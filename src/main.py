import argparse
from model import Llama4Baseline, DistilBertBaseline, DistilBertDistiller
from utils import read_yml, plot_metric_over_epochs, save_metric_to_csv
import constants
import os
from dataloader import parse_datasets
from datetime import datetime
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from dotenv import load_dotenv
load_dotenv()

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
    model_path = os.getenv("LLAMA4_PATH_SCRATCH")


    datasets_cfg = cfg.get("datasets", {})

    # To extract the dataset name for the validation set
    val_dataset_name = next(
    (dataset["name"] for dataset in datasets_cfg if dataset["set"] == "val"),
    "unknown_val_set"
    )


    val_acc_list, val_f1_list = [],[]
    if not args.baseline : # Train the model if not baseline
        # Get alpha value from config
        alpha_cfg = model_cfg.get("alpha", {})
        start, stop, step = alpha_cfg.get("start", 0.0), alpha_cfg.get("stop", 1.0), alpha_cfg.get("step", 0.1)
        #alpha_values = np.arange(start,stop+step/2,step) 
        alpha_values = [0.5]
        print("Alpha values: ", alpha_values)
        
        # Add memory optimization hooks
        def teacher_model_cleanup():
            for module in teacher_model.modules():
                if hasattr(module, '_hf_hook'):
                    module._hf_hook.pre_forward(module)
                    module._hf_hook.post_forward(module, torch.Tensor())

        teacher_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            local_files_only=True
        )
        teacher_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,  # Load model in half-precision
            local_files_only=True
        )
        print("Teacher model loaded", flush=True)

        val_acc_list_total_all_alphas, val_f1_list_total_all_alphas = [], []

        for alpha in alpha_values :
            print("Alpha: ", alpha, flush=True)

            loader_train, loader_val, loader_tests = parse_datasets(cfg=datasets_cfg,
                                                        data_path=datasets_path,
                                                        batch_size=model_cfg.get("batch_size", constants.BATCH_SIZE))

            print(f"Train set size: {len(loader_train.dataset)}")
            print(f"Val set size: {len(loader_val.dataset)}")

            model = DistilBertDistiller(device=constants.DEVICE) # new model each time

            n_outer_iterations = model_cfg.get("dagger_outer_loops", 3)
            epochs_per_step = model_cfg.get("n_epochs", constants.N_EPOCHS)

            val_acc_list_total_per_alpha, val_f1_list_total_per_alpha = [], []
            for iteration in range(n_outer_iterations):
                val_acc_list, val_f1_list = [], []
                print(f"=== Outer Step {iteration + 1}/{n_outer_iterations} ===", flush=True)
                val_acc_list, val_f1_list = model.fit(
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
                if cfg.get("special_training", {}) :
                    model.learn_from_student_errors(
                        train_loader=loader_train, teacher_model=teacher_model, 
                        teacher_tokenizer=teacher_tokenizer, n_variants=1, 
                        batch_size=model_cfg.get("batch_size", constants.BATCH_SIZE)
                    )

                #acc, f1 = model.evaluate(loader=loader_val) #useless
                val_acc_list_total_per_alpha.extend(val_acc_list)
                val_f1_list_total_per_alpha.extend(val_f1_list)
                #print(f"Validation Accuracy: {val_acc:.4f}, Validation F1: {val_f1:.4f}", flush=True)

                torch.save(model.model.state_dict(), constants.CHECKPOINT_DIR / f"model_{iteration}.pth")

                torch.cuda.empty_cache()

            val_acc_list_total_all_alphas.append(val_acc_list_total_per_alpha)
            val_f1_list_total_all_alphas.append(val_f1_list_total_per_alpha)

            file_path = constants.OUTPUTS_DIR / "logs.csv"
            file_exists = os.path.isfile(file_path)
            for test_name, test_loader in loader_tests.items():
                acc, f1 = model.evaluate(loader=test_loader)
                end_time = datetime.now()
                out_dict = {
                    'Start': start_time.strftime('%d/%m/%y %H:%M:%S'),
                    'End': end_time.strftime('%d/%m/%y %H:%M:%S'),
                    'Accuracy': acc,
                    'F1': f1,
                    'TestSet': test_name,
                    'CurrentAlpha': alpha,
                }
                out_dict["special_training"] = cfg.get("special_training", {})
                out_dict.update(model_cfg)
                #out_dict.update({dataset['set'] + "[split]": dataset['name'] + str(dataset['split']) for dataset in datasets_cfg})
                out_dict.update({
                    dataset['set'] + "[split]": dataset['name'] + str(dataset['split'])
                    for dataset in datasets_cfg
                    if dataset['set'] in {"train", "val", test_name}
                })
                print(out_dict)
                out_df = pd.DataFrame([out_dict])
                out_df.to_csv(file_path, index=False, mode='a', header=not file_exists)
                file_exists = True

            del model
            gc.collect()
            torch.cuda.empty_cache()
                
        val_acc_arr = np.array(val_acc_list_total_all_alphas)
        val_f1_arr = np.array(val_f1_list_total_all_alphas)   

        save_metric_to_csv(val_acc_arr, 
                        constants.OUTPUTS_DIR / str("val_accuracy_" + start_time.strftime('%d_%m_%y_%H_%M') +".csv"),
                        alpha_values)
        save_metric_to_csv(val_f1_arr, 
                        constants.OUTPUTS_DIR / str("val_f1_" + start_time.strftime('%d_%m_%y_%H_%M') +".csv"),
                        alpha_values) 
        
        plot_metric_over_epochs(val_acc_list_total_all_alphas, "Validation Accuracy", val_dataset_name, alpha_values, 
                    constants.OUTPUTS_DIR / str("val_accuracy_" + start_time.strftime('%d_%m_%y_%H_%M') +".png"))
        plot_metric_over_epochs(val_f1_list_total_all_alphas, "Validation F1 Score", val_dataset_name, alpha_values, 
                    constants.OUTPUTS_DIR / str("val_f1_" + start_time.strftime('%d_%m_%y_%H_%M') +".png"))   
            

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