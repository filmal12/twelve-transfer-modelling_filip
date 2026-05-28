import argparse
import subprocess

from analysis.mae_comparison import main as compare_mae
from models.training_positions import main as train_models
from models.team_model import main as train_team
from misc.prediction_analysis import run_analysis as analysis


def run_streamlit(app_path):
    subprocess.run(["streamlit", "run", app_path])

def main():
    parser = argparse.ArgumentParser(description="Project CLI")

    parser.add_argument(
        "command",
        choices=["app", "player_model", "team_model", "analysis", "improvement"],
        help="Which module to run"
    )

    args = parser.parse_args()

    if args.command == "app":
        run_streamlit("app/application.py")
    elif args.command == "player_model":
        train_models()
    elif args.command == "team_model":
        train_team()
    elif args.command == "analysis":
        analysis()
    elif args.command == "improvement":
        compare_mae()

if __name__ == "__main__":
    main()