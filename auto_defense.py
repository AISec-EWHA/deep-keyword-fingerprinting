import os
import sys
import time
from time import strftime
import logging
import subprocess
from logger_config import setup_logger


################ Only change here ################
MODEL_NAME = "dkf"          # IMPORTANT! pick one of: dkf, kfp, tiktok
RULE_NAME = "burstguard_window_mean"    # IMPORTANT! Change rule name (to anything you want)
CUDA_VISIBLE_DEVICES = "0"


######### Don't have to change from here #########
TIMESTAMP = strftime('%m%d_%H%M')
DEFENSE_RULE_NAME = f"{RULE_NAME}_{TIMESTAMP}"

# input (origianl traffic) output (defended traffic) 
INPUT_FOLDER_ROOT = "path/to/dataset"
OUTPUT_FOLDER_ROOT = "path/to/output"

# input (defended traffic) output (feature pkl file)
FEATURE_INPUT_FOLDER_ROOT = f"path/to/output/{DEFENSE_RULE_NAME}"
X_FEATURE_PATH = f"path/to/ddg_defense_{MODEL_NAME}_{DEFENSE_RULE_NAME}_x.pkl"
Y_FEATURE_PATH = f"path/to/ddg_defense_{MODEL_NAME}_{DEFENSE_RULE_NAME}_y.pkl"
PKL_FEATURE_PATH = f"path/to/ddg_defense_{MODEL_NAME}_{DEFENSE_RULE_NAME}.pkl"

# model result
CURRENT_DIR = os.path.dirname(__file__)
MODEL_RESULT = os.path.join(CURRENT_DIR, 'results', f"ow_{MODEL_NAME}_{DEFENSE_RULE_NAME}.txt")

if not os.path.exists(os.path.join(CURRENT_DIR, 'results')):
    os.makedirs(os.path.join(CURRENT_DIR, 'results'))

# log path for defended traffic
LOG_PATH = os.path.join(CURRENT_DIR, 'results', f"log_{MODEL_NAME}_{DEFENSE_RULE_NAME}.txt")
logger = setup_logger(LOG_PATH)

# script path
DEFENSE_SCIPT = os.path.join(CURRENT_DIR, 'defense', 'burstguard.py')

if MODEL_NAME=="dkf":
    FEATURE_SCIPT = os.path.join(CURRENT_DIR, 'feature', 'winlap.py')
    MODEL_SCIPT = os.path.join(CURRENT_DIR, 'model', 'main.py')
elif MODEL_NAME=="kfp":
    FEATURE_SCIPT = os.path.join(CURRENT_DIR, 'feature', 'RF.py')
    MODEL_SCIPT = os.path.join(CURRENT_DIR, 'model', 'k-FP.py')
elif MODEL_NAME=="tiktok":
    FEATURE_SCIPT = os.path.join(CURRENT_DIR, 'feature', 'Tik-Tok.py')
    MODEL_SCIPT = os.path.join(CURRENT_DIR, 'model', 'main.py')


def run_script(script_path, *args):
    cmd = [sys.executable, script_path] + list(args)
    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Script output:\n{result.stdout}")

        if result.stderr:
            logger.info(f"Script output:\n{result.stderr}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed with exit code {e.returncode}")
        logger.error(f"Script error output:\n{e.stderr}")
        sys.exit(e.returncode)


def main():
    logger = setup_logger(LOG_PATH)

    # if you want to run only some script, remove some scripts from the dictionary
    if MODEL_NAME=="dkf":
        scripts = [
            (DEFENSE_SCIPT, f"--rule-name={DEFENSE_RULE_NAME} --input-folder-root={INPUT_FOLDER_ROOT} --output-folder-root={OUTPUT_FOLDER_ROOT} --log-path={LOG_PATH}"),
            (FEATURE_SCIPT, f"--input-folder={FEATURE_INPUT_FOLDER_ROOT} --x-path={X_FEATURE_PATH} --y-path={Y_FEATURE_PATH} --log-path={LOG_PATH}"),
            (MODEL_SCIPT, f"--model-name={MODEL_NAME} --rule-name={DEFENSE_RULE_NAME} --gpu={CUDA_VISIBLE_DEVICES} --x-path={X_FEATURE_PATH} --y-path={Y_FEATURE_PATH} --model-result={MODEL_RESULT} --log-path={LOG_PATH}")
        ]
    elif MODEL_NAME=="kfp":
            scripts = [
            (DEFENSE_SCIPT, f"--rule-name={DEFENSE_RULE_NAME} --input-folder-root={INPUT_FOLDER_ROOT} --output-folder-root={OUTPUT_FOLDER_ROOT} --log-path={LOG_PATH}"),
            (FEATURE_SCIPT, f"--input-folder={FEATURE_INPUT_FOLDER_ROOT} --pkl-path={PKL_FEATURE_PATH} --log-path={LOG_PATH}"),
            (MODEL_SCIPT, f"--rule-name={DEFENSE_RULE_NAME} --pkl-path={PKL_FEATURE_PATH} --model-result={MODEL_RESULT} --log-path={LOG_PATH}")
        ]
    elif MODEL_NAME=="tiktok":
        scripts = [
            (DEFENSE_SCIPT, f"--rule-name={DEFENSE_RULE_NAME} --input-folder-root={INPUT_FOLDER_ROOT} --output-folder-root={OUTPUT_FOLDER_ROOT} --log-path={LOG_PATH}"),
            (FEATURE_SCIPT, f"--input-folder={FEATURE_INPUT_FOLDER_ROOT} --x-path={X_FEATURE_PATH} --y-path={Y_FEATURE_PATH} --log-path={LOG_PATH}"),
            (MODEL_SCIPT, f"--model-name={MODEL_NAME} --rule-name={DEFENSE_RULE_NAME} --gpu={CUDA_VISIBLE_DEVICES} --x-path={X_FEATURE_PATH} --y-path={Y_FEATURE_PATH} --model-result={MODEL_RESULT} --log-path={LOG_PATH}")
        ]

    for script, args in scripts:
        run_script(script, *args.split())
        time.sleep(1)


if __name__ == "__main__":
    main()
