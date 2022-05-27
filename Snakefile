configfile: "config.yaml"

CYCLEGAN_PARENT_FOLDER = "cycle"
CYCLEGAN_NAME_FOLDER = "pytorch-CycleGAN-and-pix2pix"
CYCLEGAN_FULL_PATH = CYCLEGAN_PARENT_FOLDER + "/" + CYCLEGAN_NAME_FOLDER

for text in ["ciao", "maxar"]:
    rule *text:
        message: text


rule setup_cyclegan:
    output: directory("./cycle/pytorch-CycleGAN-and-pix2pix")
    shell:
        "mkdir " + CYCLEGAN_PARENT_FOLDER
        "cd " + CYCLEGAN_PARENT_FOLDER
        "git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix " + CYCLEGAN_NAME_FOLDER
        "cd " + CYCLEGAN_NAME_FOLDER
        "conda env create -f environment.yml"
