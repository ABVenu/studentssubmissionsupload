import argparse

from config.config import stock_list
from data.data_loader import build_master_dataset, load_master
from models.trainer import tune_and_train_base
from models.fine_tuner import fine_tune_stock
from models.predictor import forecast_next_days
from updater.incremental_update import daily_incremental_update
from updater.add_stock import add_new_stock_and_retrain_base

def main():
    parser = argparse.ArgumentParser(description="Stock Trend Analysis CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("build-master")

    t = sub.add_parser("train-base")
    t.add_argument("--trials", type=int, default=10)
    t.add_argument("--tune", action="store_true")

    f = sub.add_parser("finetune")
    f.add_argument("--symbol", required=True)
    f.add_argument("--epochs", type=int, default=5)

    p = sub.add_parser("predict")
    p.add_argument("--symbol", required=True)
    p.add_argument("--days", type=int, default=5)
    p.add_argument("--use-ft", action="store_true")

    a = sub.add_parser("add-stock")
    a.add_argument("--symbol", required=True)
    a.add_argument("--retrain-base", action="store_true")
    a.add_argument("--optuna", action="store_true")

    args = parser.parse_args()

    if args.cmd == "build-master":
        build_master_dataset(stock_list)
    elif args.cmd == "train-base":
        master = load_master()
        if master is None:
            master = build_master_dataset(stock_list)
        tune_and_train_base(master, n_trials=args.trials, optuna_tune=args.tune)
    elif args.cmd == "finetune":
        fine_tune_stock(args.symbol, epochs=args.epochs)
    elif args.cmd == "predict":
        forecast_next_days(args.symbol, N=args.days, use_finetuned=args.use_ft)
    elif args.cmd == "add-stock":
        add_new_stock_and_retrain_base(args.symbol, retrain_base=args.retrain_base, optuna_tune=args.optuna)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
