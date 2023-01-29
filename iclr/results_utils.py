import os

def infer_run_number():
    runs = os.listdir(
            os.path.join(
                os.getcwd(),
                "drive",
                "MyDrive",
                "PyPSA_Africa_images",
                "iclr", 
                "code",
                "results"
            )
        )
    return len(runs) + 1