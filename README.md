## PyPSA meets Africa - WP 6 - Machine Learning Energy Infrastructure

Code to train NN models to recognize energy infrastrure and employ models.

Trained models are stores in ```./data/models/```.

Also, we provide code to create data for training, testing and creation of data to be evaluated.


### Dependencies

* **torch 1.9.0**
* **torchvision 0.10.0**
* **googledrivedownloader 0.4**
* **...**


### Google Cloud Development Setup

[Google Cloud Documentation Page](https://cloud.google.com/docs/)

#### To create an instance
1. Click the button with the three lines at the top left corner.
2. Click ```Compute Engine```
TODO: Add more steps here for the choice of image
9. Click ```Create``` and wait until it is done (this may take a few minutes).
12. Click  ```CREATE INSTANCE```  at the top of the window
13. Name the instance ```name-of-instance```
14. (Optional) Region: ```us-west1(Oregon)``` and Zone ```us-west-1b``` have the most K80s available
15. (Optional) Machine Configuration Series:```N1``` Machine type: ```2 vCPUs``` with ```7.5Gb memory```.
16. Click the drop-down for CPU and GPUs. Click on ```Add GPUs``` and then select `1` of type ```NVidia Tesla K80```.
17. Disk size should be at least 50GB. 
18. Click ```Create```.

#### To Log into the instance via terminal
2. Download the `gcloud` toolkit using ```curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-269.0.0-linux-x86_64.tar.gz```
3. Install the `gcloud` toolkit using ```tar zxvf google-cloud-sdk-269.0.0-linux-x86_64.tar.gz; bash google-cloud-sdk/install.sh```.
**Note**: You will be asked to provide a passphrase to generate your local key, simply use a password of your choice. There might be some Yes/No style questions as well, choose yes, when that happens.
4. Reset terminal using ```reset; source ~/.bashrc```. Then authorize the current machine to access your nodes run ```gcloud auth login```. This will authenticate your google account login.
3. Follow the prompts to get a token for your current machine.
4. Run ```gcloud config set project PROJECT_ID``` where you replace `PROJECT-ID` with your project ID, you can find that in the projects drop down menu on the top of the Google Compute Engine window; this sets the current project as the active one
5. In your compute engine window, in the line for the instance  that you have started (`name-of-instance`), click on the downward arrow next to ```ssh```. Choose ```View gcloud command```. Copy the command to your terminal and press enter.
6. Add a password for your ssh-key. Re-enter password when prompted.
11. Then clone the repo

##### Known Errors
If your **instance failed to create due to the following error** - ```Quota 'GPUS_ALL_REGIONS' exceeded. Limit: 0.0 globally.```:
20. Click the button with the three lines at the top left corner.
21. Go to ```IAM & Admin -> Quotas```.
22. In the Filter Table, type GPU and Press Enter. Then Go to ```GPUs (all regions)``` and click ```All Quotas```.
23. Tick in the box next to Global and then Click ```Edit Quotas``` in the top bar. 
24. This will open a box in the right side corner asking for your details. Fill in those and then click Next.
25. Put your New Limit as ```1``` and in the description you can mention the reason. And then Send Request. 
26. You will receive a confirmation email with your Quota Limit increased. This may take some minutes.
27. After the confirmation email, you can recheck the GPU(All Regions) Quota Limit being set to 1. This usually shows up in 10-15 minutes after the confirmation email. 
28. Retry making the VM instance again as before and you should have your instance now. 

On the **first login** you may see an error of the form `Unable to set persistence mode for GPU 00000000:00:04.0: Insufficient Permissions`
Ignore this.  The instance on the first startup checks for the gpu cuda drivers and since they are not there, it will install them. This will only happen once on your first login.


##### Training the model 
To test PyTorch running on the GPU, run the test train script (to be created or added as an argument).
```
python train.py
```

##### Note
Any other things that are useful for development goe here

$50 dollars worth of credit should be about 125 hours of GPU usage on a K80.

Run ```nvidia-smi``` to confirm that the GPU can be found.

**Remember** to ```stop``` the instance when not in use:
To stop the instance go to `Compute Engine -> VM instances` on the Google Cloud Platform, slect the instance and click ```Stop```.

**Future ssh access** `gcloud` command copied from the google compute engine instance page

**Copying data to and from an instance**
[Google Docs Page On Copying Data](https://cloud.google.com/filestore/docs/copying-data).
[Stackoverflow post](https://stackoverflow.com/questions/27857532/rsync-to-google-compute-engine-instance-from-jenkins).

**Screen Commands**:
```screen```
```screen -ls```
```screen -d -r screen_id```  Replacing screen_id 

While in a session,
- ```ctrl+a+esc``` to pause process and be able to scroll
- ```ctrl+a+d``` to detach from session (reattach using ```screen -r```)
- ```ctrl+a+n``` to see the next session
- ```ctrl+a+c``` to create a new session


### University of Edinburgh Informatics Cluster

```
ssh [student-number]@student.ssh.inf.ed.ac.uk
```
```
ssh student.compute
```
### Alternative Google Cloud Platform setup
Not focused on Machine Learning setup, however, might be complementary to the above description.
[PyPSA Google Cloud Documentation Page](https://pypsa-eur.readthedocs.io/en/latest/cloudcomputing.html#cloud-computing)
