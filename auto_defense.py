import os
import sys
import time
from time import strftime
import logging
import subprocess

################ Only change here ################
RULE_NAME = "burstguard"    # IMPORTANT! Change rule name
CUDA_VISIBLE_DEVICES = "1"


######### Don't have to change from here #########
TIMESTAMP = strftime('%m%d_%H%M')
DEFENSE_RULE_NAME = f"{RULE_NAME}_{TIMESTAMP}"

# input (origianl traffic) output (defended traffic) 
INPUT_FOLDER_ROOT = "/scratch3/KF/duckduckgo/new_processed_to_txt_273_1000_unmerged"
OUTPUT_FOLDER_ROOT = "/scratch2/DKF/BurstGuard/defended"

# input (defended traffic) output (feature pkl file)
FEATURE_INPUT_FOLDER_ROOT = f"/scratch2/DKF/BurstGuard/defended/{DEFENSE_RULE_NAME}"
X_FEATURE_PATH = f"/scratch2/DKF/BurstGuard/feature/ddg_defense_{DEFENSE_RULE_NAME}_y.pkl"
Y_FEATURE_PATH = f"/scratch2/DKF/BurstGuard/feature/ddg_defense_{DEFENSE_RULE_NAME}_X.pkl"

# model result
CURRENT_DIR = os.path.dirname(__file__)
MODEL_RESULT = os.path.join(CURRENT_DIR, 'results', f"{DEFENSE_RULE_NAME}.txt")

if not os.path.exists(os.path.join(CURRENT_DIR, 'results')):
    os.makedirs(os.path.join(CURRENT_DIR, 'results'))

# log path for defended traffic
LOG_PATH = os.path.join(CURRENT_DIR, 'results', f"log_{DEFENSE_RULE_NAME}.txt")

# script path
DEFENSE_SCIPT = os.path.join(CURRENT_DIR, 'defense', 'burstguard.py')
FEATURE_SCIPT = os.path.join(CURRENT_DIR, 'feature', 'winlap.py')
MODEL_SCIPT = os.path.join(CURRENT_DIR, 'model', 'main.py')


def run_script(script_path, *args):
    cmd = [sys.executable, script_path] + list(args)
    logging.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logging.info(f"Script output:\n{result.stdout}")

        if result.stderr:
            logging.error(f"Script error output:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Script failed with exit code {e.returncode}")
        logging.error(f"Script error output:\n{e.stderr}")
        sys.exit(e.returncode)


def main():
    # if you want to run only some script, remove some scripts from the dictionary
    scripts = [
        (DEFENSE_SCIPT, f"--rule-name={DEFENSE_RULE_NAME} --input-folder-root={INPUT_FOLDER_ROOT} --output-folder-root={OUTPUT_FOLDER_ROOT} --log-path={LOG_PATH}"),
        (FEATURE_SCIPT, f"--input-folder={FEATURE_INPUT_FOLDER_ROOT} --x-path={X_FEATURE_PATH} --y-path={Y_FEATURE_PATH}"),
        (MODEL_SCIPT, f"--rule-name={DEFENSE_RULE_NAME} --gpu={CUDA_VISIBLE_DEVICES} --x-path={X_FEATURE_PATH} --y-path={Y_FEATURE_PATH} --model-result={MODEL_RESULT}")
    ]

    for script, args in scripts:
        run_script(script, *args.split())
        time.sleep(1)


if __name__ == "__main__":
    main()