# -*- coding: utf-8 -*-
import os
import socket
import time
import traceback
import uuid

import psutil
import torch
import wandb
from absl import app, flags
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger

import configs
from models.train_loop import TrainLoop
from utils import logger
from utils.functional import console_params
from utils.io import format_yaml, load_yaml
from utils.paths import get_models_checkpoint_path

# disable parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="config",
    short_name="c",
    default=None,
    required=True,
    help="Setting config path",
)
flags.DEFINE_bool(
    name="use_wandb",
    default=False,
    help="Using wandb logger.",
)

flags.DEFINE_string(
    name="experiment_id",
    default=None,
    help="Setting experiment id.",
)
flags.DEFINE_string(
    name="weight_name",
    default="last.ckpt",
    help="Setting model weight name when resume to train.",
)

flags.DEFINE_enum(
    name="type",
    default="all",
    enum_values=["all", "fit", "test"],
    help="Setting running type.",
)

flags.DEFINE_bool(
    name="skip_sanity_valid",
    default=False,
    short_name="s",
    help="Skipping trainer sanity valid step.",
)

flags.DEFINE_bool(
    name="shutdown",
    default=False,
    help="Shutdown server when the run ends.",
)

flags.DEFINE_bool(
    name="early_stopping",
    default=True,
    help="Using early stopping callbacks.",
)

flags.DEFINE_integer(
    name="patience", default=5, help="Setting EarlyStop callback `patience` param."
)

flags.DEFINE_enum(
    name="checkpoint_mode",
    default="max",
    enum_values=["max", "min"],
    help="Setting running type.",
)

flags.DEFINE_enum(
    name="checkpoint_monitor",
    default="acc",
    enum_values=["acc", "f1_macro", "f1_micro", "loss"],
    help="Setting running type.",
)


def main(argv):
    del argv

    # main process ID
    main_process_id = os.getpid()

    # load config and update config from console params
    config = load_yaml(FLAGS.config, return_str=False)
    config.update(console_params())
    logger.info(f"Print Config Info:\n{format_yaml(config)}")

    # random seed
    seed_everything(config.base.global_seed)
    torch.set_float32_matmul_precision("medium")

    if FLAGS.experiment_id is None:
        experiment_id = (
            wandb.wandb_lib.runid.generate_id()
            if FLAGS.use_wandb
            else str(uuid.uuid4())[:8]
        )
    else:
        experiment_id = FLAGS.experiment_id

    model_name = (
        f"{'debug' if FLAGS.log_level == 'debug' else 'model'}"
        f"-{config.dataset.name}"
        f"-{config.dynamic.loss_scaler}"
        f"-{config.optimizer.name}"
        f"-{config.optimizer.lr}"
        f"-{config.optimizer.scheduler_name}"
        f"-{socket.gethostname()[:8]}"
    )

    # config logger
    wandb_logger = (
        WandbLogger(
            project=config.base.project_name,
            name=f"{model_name}-{experiment_id}",
            config=config,
            resume="allow",
            id=experiment_id,
        )
        if FLAGS.use_wandb
        else None
    )

    trainer = init_trainer(config, experiment_id, model_name, wandb_logger)
    model = TrainLoop(
        config,
        is_dist=(trainer.num_devices * trainer.num_nodes) > 1,
        wandb_logger=wandb_logger,
    )
    if wandb_logger is not None:
        wandb_logger.watch(model, log="all")
    ckpt_path = None
    if FLAGS.experiment_id is not None:
        ckpt_path = os.path.join(
            get_models_checkpoint_path(experiment_id), FLAGS.weight_name
        )

    try:
        trained = False
        if FLAGS.type in ["all", "fit"]:
            trainer.fit(model=model, ckpt_path=ckpt_path)
            trained = True

        if FLAGS.type in ["all", "test"]:
            if trained:
                trainer.test(ckpt_path="best")
            else:
                trainer.test(model=model, ckpt_path=ckpt_path)
    except Exception as e:
        logger.error(f"Exception Caused: {e}")
        logger.error(f"Function Stack: \n{traceback.format_exc()}")
    finally:
        if main_process_id == os.getpid():
            # shutdown server
            shutdown_server(shutdown=FLAGS.shutdown)


def wait_wandb():
    children = psutil.Process().children()
    wandb_process = None
    for child in children:
        if "wandb-service" in child.name():
            wandb_process = child

    # unused wandb service
    if wandb_process is None:
        return

    sleep_time = 0
    max_sleep_time = 60 * 5
    while wandb_process.status() not in [
        psutil.STATUS_ZOMBIE,
        psutil.STATUS_DEAD,
    ]:
        # wait wandb service to finish
        time.sleep(5)

        # set max wait time: 5min
        sleep_time += 5
        if sleep_time >= max_sleep_time:
            break


def init_trainer(
    config: configs.Config,
    experiment_id: str,
    model_name: str,
    wandb_logger: WandbLogger,
) -> Trainer:
    monitor_name = f"{FLAGS.checkpoint_monitor}_epoch_valid"
    # checkpoint callback
    model_checkpoint = ModelCheckpoint(
        dirpath=get_models_checkpoint_path(experiment_id),
        filename=model_name + "{epoch:02d}_{" + f"{monitor_name}" + ":.4f}",
        mode=FLAGS.checkpoint_mode,
        monitor=monitor_name,
        save_last=True,
        save_top_k=config.trainer.save_top_k,
    )

    # params
    trainer_config = config.trainer
    if FLAGS.skip_sanity_valid:
        trainer_config.num_sanity_val_steps = 0

    params = config.trainer.__dict__
    params.pop("save_top_k")

    # setting devices and num_nodes to 1, when multi devices hosts
    # avoid multi devices to repeat samples of test dataset.
    if FLAGS.type == "test":
        params["devices"] = 1
        params["num_nodes"] = 1

    # debug params
    if FLAGS.log_level == "debug":
        debug_params = {
            "limit_train_batches": 0.1,
            "limit_val_batches": 0.2,
            "limit_test_batches": 0.2,
            "limit_predict_batches": 0.2,
            "check_val_every_n_epoch": 1,
            "max_epochs": 2,
        }
        # update trainer params
        params.update(debug_params)

    # callbacks list
    callbacks = [
        model_checkpoint,
        RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]

    if FLAGS.early_stopping:
        # early stop callback
        early_stopping = EarlyStopping(
            monitor=monitor_name,
            patience=FLAGS.patience,
            verbose=True,
            mode=FLAGS.checkpoint_mode,
            min_delta=0.0003,
        )
        callbacks.append(early_stopping)

    return Trainer(
        callbacks=callbacks,
        logger=wandb_logger,
        **params,
    )


def shutdown_server(shutdown: bool = False):
    if not shutdown:
        return
    wait_wandb()
    os.system("shutdown")


if __name__ == "__main__":
    app.run(main)
