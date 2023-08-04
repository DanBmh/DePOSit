import copy
import os
import tqdm
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam

from model import ModelMain

import sys

sys.path.append("/PoseForecaster/")
import utils_pipeline

# ==================================================================================================

datamode = "gt-gt"
# datamode = "pred-gt"
# datamode = "pred-pred"

config_sk = {
    # "item_step": 2,
    # "window_step": 2,
    "item_step": 1,
    "window_step": 1,
    "select_joints": [
        "hip_middle",
        "hip_right",
        "knee_right",
        "ankle_right",
        "hip_left",
        "knee_left",
        "ankle_left",
        "nose",
        "shoulder_left",
        "elbow_left",
        "wrist_left",
        "shoulder_right",
        "elbow_right",
        "wrist_right",
        "shoulder_middle",
    ],
}

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_kppspose_10fps.json",
    # "/datasets/preprocessed/human36m/train_forecast_kppspose.json",
]

# datasets_train = [
#     "/datasets/preprocessed/mocap/train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlmovi_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/bmlrub_train_forecast_samples_10fps.json",
#     "/datasets/preprocessed/amass/kit_train_forecast_samples_10fps.json"
# ]

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_kppspose.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_10fps.json"
# dataset_eval_test = "/datasets/preprocessed/mocap/{}_forecast_samples_4fps.json"

num_joints = len(config_sk["select_joints"])
in_features = num_joints * 3
dim_used = list(range(in_features))

# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device):
    sequences = utils_pipeline.make_input_sequence(batch, split, "gt-gt")

    # Convert to meters
    sequences = sequences / 1000.0

    return sequences


def prepare_batch(sequences_train, sequences_gt, batch_size: int, device):
    # Merge joints and coordinates to a single dimension
    sequences_train = sequences_train.reshape([batch_size, sequences_train.shape[1], -1])
    sequences_gt = sequences_gt.reshape([batch_size, sequences_gt.shape[1], -1])

    all_seq = np.concatenate([sequences_train, sequences_gt], axis=1)

    mask1 = np.ones_like(sequences_train)
    mask2 = np.zeros_like(sequences_gt)
    mask = np.concatenate([mask1, mask2], axis=1)

    timepoints = np.array([list(range(all_seq.shape[1]))])
    timepoints = np.repeat(timepoints, batch_size, axis=0)

    all_seq = torch.from_numpy(all_seq).to(device)
    mask = torch.from_numpy(mask).to(device)
    timepoints = torch.from_numpy(timepoints).to(device)

    batch = {
        "pose": all_seq,
        "mask": mask,
        "timepoints": timepoints,
    }

    return batch


# ==================================================================================================


parser = argparse.ArgumentParser(description="Arguments for running the scripts")
parser.add_argument("--miss_rate", type=int, default=20)
parser.add_argument(
    "--miss_type",
    type=str,
    default="no_miss",
    choices=[
        "no_miss",
        "random",
        "random_joints",
        "random_right_leg",
        "random_left_arm_right_leg",
        "structured_joint",
        "structured_frame",
        "random_frame",
        "noisy_25",
        "noisy_50",
    ],
    help="Choose the missing type of input sequence",
)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--skip_rate_train", type=int, default=1)
parser.add_argument("--skip_rate_val", type=int, default=25)
parser.add_argument("--joints", type=int, default=32)
parser.add_argument("--input_n", type=int, default=25)
parser.add_argument("--output_n", type=int, default=25)
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=["train", "test"],
    help="Choose to train or test from the model",
)
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    "--data",
    type=str,
    default="all",
    choices=["one", "all"],
    help="Choose to train on one subject or all",
)
parser.add_argument("--output_dir", type=str, default="default")
parser.add_argument("--model_s", type=str, default="default")
parser.add_argument("--model_l", type=str, default="default")
parser.add_argument("--data_dir", type=str, default="/datasets/")
parser.add_argument(
    "--model_weights_path",
    type=str,
    default="",
    help="directory with the model weights to copy",
)

args = parser.parse_args()
print(args)

config_sk["input_n"] = args.input_n
config_sk["output_n"] = args.output_n

config_dp = {
    "train": {
        "epochs": 50,
        "batch_size": 32,
        "batch_size_test": 1,
        "lr": 1.0e-3,
    },
    "diffusion": {
        "layers": 12,
        "channels": 64,
        "nheads": 8,
        "diffusion_embedding_dim": 128,
        "beta_start": 0.0001,
        "beta_end": 0.5,
        "num_steps": 50,
        "schedule": "cosine",
    },
    "model": {
        "is_unconditional": 0,
        "timeemb": 128,
        "featureemb": 16,
    },
}


def save_csv_log(head, value, is_create=False, file_name="test"):
    if len(value.shape) < 2:
        value = np.expand_dims(value, axis=0)
    df = pd.DataFrame(value)
    file_path = f"{output_dir}/{file_name}.csv"
    if not os.path.exists(file_path) or is_create:
        df.to_csv(file_path, header=head, index=False)
    else:
        with open(file_path, "a") as f:
            df.to_csv(f, header=False, index=False)


def save_state(model, optimizer, scheduler, epoch_no, foldername):
    params = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch_no,
    }
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), foldername + "/model.pth")
    else:
        torch.save(model.state_dict(), foldername + "/model.pth")
    torch.save(params, foldername + "/params.pth")


def train(
    model,
    config_dp,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
    load_state=False,
    dlen_train=0,
    dlen_eval=0,
):
    optimizer = Adam(model.parameters(), lr=config_dp["lr"], weight_decay=1e-6)
    if load_state:
        optimizer.load_state_dict(torch.load(f"{output_dir}/params.pth")["optimizer"])

    p1 = int(0.75 * config_dp["epochs"])
    p2 = int(0.9 * config_dp["epochs"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    if load_state:
        lr_scheduler.load_state_dict(
            torch.load(f"{output_dir}/params.pth")["scheduler"]
        )

    train_loss = []
    valid_loss = []
    train_loss_epoch = []
    valid_loss_epoch = []

    best_valid_loss = 1e10
    start_epoch = 0
    if load_state:
        start_epoch = torch.load(f"{output_dir}/params.pth")["epoch"]

    for epoch_no in range(start_epoch, config_dp["epochs"]):
        avg_loss = 0
        model.train()

        label_gen_train = utils_pipeline.create_labels_generator(
            train_loader, config_sk
        )
        label_gen_eval = utils_pipeline.create_labels_generator(valid_loader, config_sk)

        nbatch = config_dp["batch_size"]
        batch_no = 0
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen_train, batch_size=nbatch),
            total=int(dlen_train / nbatch),
        ):
            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)

            augment = True
            if augment:
                sequences_train, sequences_gt = utils_pipeline.apply_augmentations(
                    sequences_train, sequences_gt
                )

            batch = prepare_batch(sequences_train, sequences_gt, nbatch, device)
            batch_no += 1

            # print(batch)
            # print(batch["pose"].shape)
            # print(batch["mask"].shape)
            # print(batch["timepoints"].shape)
            # exit()

            optimizer.zero_grad()
            loss = model(batch).mean()
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

            # break

        lr_scheduler.step()
        train_loss.append(avg_loss / batch_no)
        train_loss_epoch.append(epoch_no)

        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            print("Validating ...")

            with torch.no_grad():
                nbatch = config_dp["batch_size"]
                for batch in tqdm.tqdm(
                    utils_pipeline.batch_iterate(label_gen_eval, batch_size=nbatch),
                    total=int(dlen_eval / nbatch),
                ):
                    sequences_train = prepare_sequences(batch, nbatch, "input", device)
                    sequences_gt = prepare_sequences(batch, nbatch, "target", device)
                    batch = prepare_batch(sequences_train, sequences_gt, nbatch, device)

                    loss = model(batch, is_train=0).mean()
                    avg_loss_valid += loss.item()

                    # break

            valid_loss.append(avg_loss_valid / batch_no)
            valid_loss_epoch.append(epoch_no)
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                save_state(model, optimizer, lr_scheduler, epoch_no, foldername)

            if (epoch_no + 1) == config_dp["epochs"]:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(train_loss_epoch, train_loss)
                ax.plot(valid_loss_epoch, valid_loss)
                ax.grid(True)
                plt.show()
                fig.savefig(f"{foldername}/loss.png")

    save_state(model, optimizer, lr_scheduler, config_dp["epochs"], foldername)
    np.save(f"{foldername}/train_loss.npy", np.array(train_loss))
    np.save(f"{foldername}/valid_loss.npy", np.array(valid_loss))


def mpjpe_error(batch_imp, batch_gt):
    batch_imp = batch_imp.contiguous().view(-1, 3)
    batch_gt = batch_gt.contiguous().view(-1, 3)

    return torch.mean(torch.norm(batch_gt - batch_imp, 2, 1))


def evaluate(
    model, label_gen_test, nsample=5, scaler=1, sample_strategy="best", dlen_test=0
):
    with torch.no_grad():
        model.eval()
        mpjpe_total = 0

        all_target = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        titles = np.array(range(output_n)) + 1
        m_p3d_h36 = np.zeros([output_n])
        n = 0

        nbatch = config_dp["train"]["batch_size_test"]
        batch_no = 0
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen_test, batch_size=nbatch),
            total=int(dlen_test / nbatch),
        ):
            n += nbatch
            batch_no += 1

            sequences_train = prepare_sequences(batch, nbatch, "input", device)
            sequences_gt = prepare_sequences(batch, nbatch, "target", device)
            batch = prepare_batch(sequences_train, sequences_gt, nbatch, device)

            if isinstance(model, nn.DataParallel):
                output = model.module.evaluate(batch, nsample)
            else:
                output = model.evaluate(batch, nsample)

            gt = batch["pose"].clone()

            samples, c_target, eval_points, observed_time = output
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = gt  # (B,L,K)
            eval_points = eval_points.permute(0, 2, 1)

            samples_mean = np.mean(samples.cpu().numpy(), axis=1)

            renorm_pose = []
            renorm_c_target = []
            renorm_all_joints_seq = []

            for i in range(len(samples_mean)):
                renorm_c_target_i = c_target.cpu().data.numpy()[i][input_n:] * 1000

                if sample_strategy == "best":
                    best_renorm_pose = None
                    best_error = float("inf")

                    for j in range(nsample):
                        renorm_pose_j = samples.cpu().numpy()[i][j][input_n:] * 1000
                        error = mpjpe_error(
                            torch.from_numpy(renorm_pose_j).view(
                                output_n, num_joints, 3
                            ),
                            torch.from_numpy(renorm_c_target_i).view(
                                output_n, num_joints, 3
                            ),
                        )
                        if error.item() < best_error:
                            best_error = error.item()
                            best_renorm_pose = renorm_pose_j
                else:
                    best_renorm_pose = samples_mean[i][input_n:] * 1000
                renorm_pose.append(best_renorm_pose)
                renorm_c_target.append(renorm_c_target_i)

            renorm_pose = torch.from_numpy(np.array(renorm_pose))
            renorm_c_target = torch.from_numpy(np.array(renorm_c_target))

            mpjpe_p3d_h36 = torch.sum(
                torch.mean(
                    torch.norm(
                        renorm_c_target.view(-1, output_n, num_joints, 3)
                        - renorm_pose.view(-1, output_n, num_joints, 3),
                        dim=3,
                    ),
                    dim=2,
                ),
                dim=0,
            )
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()

            all_target.append(renorm_c_target)
            all_evalpoint.append(eval_points)
            all_observed_time.append(observed_time)
            all_generated_samples.append(renorm_all_joints_seq)

            mpjpe_current = mpjpe_error(
                renorm_pose.view(-1, output_n, num_joints, 3),
                renorm_c_target.view(-1, output_n, num_joints, 3),
            )
            mpjpe_total += mpjpe_current.item()

            if batch_no == 100:
                break

        print("Average MPJPE:", mpjpe_total / batch_no)

        ret = {}
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(output_n):
            ret["#{:d}".format(titles[j])] = m_p3d_h36[j]

        return all_generated_samples, all_target, all_evalpoint, ret


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: %s" % device)

    data_dir = args.data_dir
    output_dir = f"{args.output_dir}"
    input_n = args.input_n
    output_n = args.output_n
    skip_rate = args.skip_rate_train
    config_dp["train"]["epochs"] = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.mode == "train":
        model = ModelMain(config_dp, device, target_dim=in_features)

        if args.model_weights_path != "":
            print("Loading model weights from:", args.model_weights_path)
            model.load_state_dict(torch.load(args.model_weights_path), strict=False)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        model.to(device)

        # Load preprocessed datasets
        print("Loading datasets ...")
        dataset_train, dlen_train = [], 0
        for dp in datasets_train:
            cfg = copy.deepcopy(config_sk)
            if "mocap" in dp:
                cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"

            ds, dlen = utils_pipeline.load_dataset(dp, "train", cfg)
            dataset_train.extend(ds["sequences"])
            dlen_train += dlen
        esplit = "test" if "mocap" in dataset_eval_test else "eval"
        cfg = copy.deepcopy(config_sk)
        if "mocap" in dataset_eval_test:
            cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
        dataset_eval, dlen_eval = utils_pipeline.load_dataset(
            dataset_eval_test, esplit, cfg
        )
        dataset_eval = dataset_eval["sequences"]

        # dataset_train, dlen_train = utils_pipeline.load_dataset(
        #     datapath_preprocessed, "eval", config_sk
        # )
        # dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        #     datapath_preprocessed, "eval", config_sk
        # )

        train(
            model,
            config_dp["train"],
            dataset_train,
            valid_loader=dataset_eval,
            foldername=output_dir,
            load_state=args.resume,
            dlen_train=dlen_train,
            dlen_eval=dlen_eval,
        )

    elif args.mode == "test":
        model = ModelMain(config_dp, device, target_dim=in_features)
        print(
            ">>> total params: {:.2f}M".format(
                sum(p.numel() for p in model.parameters()) / 1000000.0
            )
        )

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        cfg = copy.deepcopy(config_sk)
        if "mocap" in dataset_eval_test:
            cfg["select_joints"][cfg["select_joints"].index("nose")] = "head_upper"
        dataset_test, dlen_test = utils_pipeline.load_dataset(
            dataset_eval_test, "test", cfg
        )
        dataset_test = dataset_test["sequences"]
        # dataset_test, dlen_test = utils_pipeline.load_dataset(
        #     datapath_preprocessed, "test", config_sk
        # )

        label_gen_test = utils_pipeline.create_labels_generator(dataset_test, config_sk)

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(torch.load(f"{args.model_l}/model.pth"))
        else:
            model.load_state_dict(torch.load(f"{args.model_l}/model.pth"))

        head = np.array(["act"])
        for k in range(1, output_n + 1):
            head = np.append(head, [f"#{k}"])
        errs = np.zeros([1 + 1, output_n])

        pose, target, mask, ret = evaluate(
            model,
            label_gen_test,
            nsample=5,
            scaler=1,
            sample_strategy="best",
            dlen_test=dlen_test,
        )

        ret_log = np.array([])
        for k in ret.keys():
            ret_log = np.append(ret_log, [ret[k]])
        errs[0] = ret_log

        errs[-1] = np.mean(errs[:-1], axis=0)
        actions = np.expand_dims(np.array(["all"] + ["average"]), axis=1)
        value = np.concatenate([actions, errs.astype(np.str)], axis=1)
        save_csv_log(head, value, is_create=True, file_name="fde_per_action")
