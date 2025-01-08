import os
import sys
import time
from time import strftime
import logging
import subprocess
from logger_config import setup_logger

# NOTE:
# 1. auto_defense.py 파일에서, RULE_NAME (원하는 대로)과 CUDA_VISIBLE_DEVICES를 변경해주세요.
# 2. /defense/burstguard.py 파일에서, 새로운 defense 룰을 새로운 함수로 만들어주세요. 이 함수는 Packet 객체들의 리스트 (즉, 하나의 trace)를 input으로 받습니다.
#   * Packet 객체는 timestamp, direction, length, dummy를 속성으로 가지고 있습니다.
# 3. 위에서 추가한 새로운 함수를 적용하기 위해서 file_read_write 함수에서, simulated = 새로운함수이름(trace) 로 변경해주세요.
# 4. 그 후 python3 auto_defense.py 명령어를 입력하면, `defense 적용 -> defended traffic에서 winlap feature 추출 -> 해당 feature로 DKF 모델 결과 얻기`가 전부 run 됩니다.

# * defended traffic 및 feature는 /scratch2/DKF/BurstGuard 경로에, bandwidth overhead 및 모델 정확도 로그는 /results 경로에 저장됩니다.


################ Only change here ################
MODEL_NAME = "kfp"          # IMPORTANT! pick one of: dkf, kfp, tiktok
RULE_NAME = "burstguard_window_mean"    # IMPORTANT! Change rule name
CUDA_VISIBLE_DEVICES = "6"


######### Don't have to change from here #########
TIMESTAMP = strftime('%m%d_%H%M')
DEFENSE_RULE_NAME = f"{RULE_NAME}_{TIMESTAMP}"

# input (origianl traffic) output (defended traffic) 
INPUT_FOLDER_ROOT = "/scratch3/KF/duckduckgo/new_processed_to_txt_273_1000_unmerged"
OUTPUT_FOLDER_ROOT = "/scratch2/DKF/BurstGuard/defended"

# input (defended traffic) output (feature pkl file)
FEATURE_INPUT_FOLDER_ROOT = f"/scratch2/DKF/BurstGuard/defended/{DEFENSE_RULE_NAME}"
X_FEATURE_PATH = f"/scratch2/DKF/BurstGuard/feature/ddg_defense_{MODEL_NAME}_{DEFENSE_RULE_NAME}_x.pkl"
Y_FEATURE_PATH = f"/scratch2/DKF/BurstGuard/feature/ddg_defense_{MODEL_NAME}_{DEFENSE_RULE_NAME}_y.pkl"
PKL_FEATURE_PATH = f"/scratch2/DKF/BurstGuard/feature/ddg_defense_{MODEL_NAME}_{DEFENSE_RULE_NAME}.pkl"

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