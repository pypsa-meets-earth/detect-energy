configfile: "config.yaml"
configfile: "datasets/dataset_config.yaml"

CYCLEGAN_PARENT_FOLDER = "cycle"
CYCLEGAN_NAME_FOLDER = "pytorch-CycleGAN-and-pix2pix"
CYCLEGAN_FULL_PATH = CYCLEGAN_PARENT_FOLDER + "/" + CYCLEGAN_NAME_FOLDER

wildcard_constraints:
    dataset: "|".join(config["datasets"].keys())

rule setup_cyclegan:
    output: directory(CYCLEGAN_FULL_PATH)
    shell:
        "mkdir " + CYCLEGAN_PARENT_FOLDER
        "cd " + CYCLEGAN_PARENT_FOLDER
        "git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix " + CYCLEGAN_NAME_FOLDER
        "cd " + CYCLEGAN_NAME_FOLDER
        "conda env create -f environment.yml"

rule download_dataset_gdrive:
    output: directory("datasets/{dataset}")
    message:
        "Downloading the dataset implies accepting the data licencing by authors"
        "Requires"
    shell:
        "gdrive download --path datasets/ " + config["datasets"]["{dataset}"]["gdrive"]
