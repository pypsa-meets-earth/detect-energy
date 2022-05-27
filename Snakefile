import sys
import os

sys.path.append("./scripts")

configfile: "config.yaml"
configfile: "datasets/dataset_config.yaml"

CYCLEGAN_PARENT_FOLDER = "cycle"
CYCLEGAN_NAME_FOLDER = "pytorch-CycleGAN-and-pix2pix"
CYCLEGAN_FULL_PATH = os.path.abspath(CYCLEGAN_PARENT_FOLDER + "/" + CYCLEGAN_NAME_FOLDER)

wildcard_constraints:
    preload_dataset="|".join(config["datasets"].keys()),
    general_dataset="[-+a-zA-Z0-9\.]*",

# rule to setup the cyclegan repo
rule setup_cyclegan:
    output: directory(CYCLEGAN_FULL_PATH)
    run:
        shell("mkdir -p " + os.path.abspath(CYCLEGAN_PARENT_FOLDER))
        shell("git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix " + CYCLEGAN_FULL_PATH)
        shell("pip install -r " + CYCLEGAN_FULL_PATH + "/requirements.txt")

# Rule to download data from google drive
rule download_dataset_gdrive:
    output: directory("datasets/{preload_dataset}")
    message:
        "Downloading the dataset implies accepting the data licencing by authors"
        "Requires"
    run:
        shell("gdrive download --path datasets/ " + config["datasets"]["{preload_dataset}"]["gdrive"])


rule cycle_train:
    input:
        cyclegan_dir=directory(CYCLEGAN_FULL_PATH),
        training_dataset=directory("datasets/{general_dataset}"),
    run:
        shell("cd " + CYCLEGAN_FULL_PATH)
        shell("python train.py --dataroot " + os.path.abspath(input["training_dataset"]) + "--name {general_dataset} --model cycle_gan")


rule cycle_test:
    input:
        cyclegan_dir=directory(CYCLEGAN_FULL_PATH),
        training_dataset=directory("datasets/{general_dataset}"),
    run:
        shell("cd " + CYCLEGAN_FULL_PATH)
        shell("python train.py --dataroot " + os.path.abspath(input["training_dataset"]) + "--name {general_dataset} --no_dropout --model cycle_gan --direction BtoA")