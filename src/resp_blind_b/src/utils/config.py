import argparse
import yaml
import sys
import copy

from src.utils.resources import OUTPUT_DIR, get_system_prompt

def parse_config(config_path: str = "config/config.yaml"):
    with open("config/models.yaml", "r") as f:
        models = yaml.safe_load(f)

    with open(config_path, "r") as f:
        base_cfg = yaml.safe_load(f)
        
    parser = argparse.ArgumentParser(description="Response Generation Configuration")
    parser.add_argument("--name", "-n", type=str, required=True, help="Name of the generation run (used for output directory)")
    parser.add_argument("--model_name", "-m", type=str, default="gemma-26B", help="Name of the model to use (must be defined in models.yaml)")
    parser.add_argument("--prompt", "-p", type=str, default="baseline", help="Name of the system prompt to use")
    parser.add_argument("--additional_args", "-a", action="append", help="Additional key=value pairs to override config values")
    args = parser.parse_args()

    if args.model_name not in models:
        raise ValueError(f"Model '{args.model_name}' not found in models.yaml. Available models: {list(models.keys())}")
    
    model_cfg = models[args.model_name]
    system_prompt = get_system_prompt(args.prompt)
    output_dir = OUTPUT_DIR / args.name
    output_dir.mkdir(exist_ok=True, parents=True)
    executed_script = sys.argv[0]

    cfg = {
        **base_cfg, 
        "model": model_cfg, 
        "prompt": system_prompt, 
        "output_dir": output_dir,
        "executed_script": executed_script
    }
    
    if args.additional_args:
        for arg in args.additional_args:
            if "=" not in arg:
                raise ValueError(f"Invalid argument format: '{arg}'. Must be 'key=value'.")
            
            keys_str, value = arg.split("=", 1)
            keys = keys_str.split(".")
            
            # Navigate through the config dictionary to set the value at the correct nested level
            d = cfg
            for key in keys[:-1]:
                if key not in d:
                    d[key] = {}
                d = d[key]
            
            # Attempt to parse value as int, float, or bool, otherwise keep as string
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            
            d[keys[-1]] = value

    cfg_to_save = copy.deepcopy(cfg)
    cfg_to_save["output_dir"] = str(cfg_to_save["output_dir"])
    
    config_file_path = output_dir / "run_config.yaml"
    with open(config_file_path, "w") as f:
        yaml.dump(cfg_to_save, f, default_flow_style=False, sort_keys=False)

    return cfg