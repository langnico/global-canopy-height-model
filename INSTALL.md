## Installation and credentials

Here we present two ways how to install the packages. 
- A) requires GDAL to be installed on the system first. 
- B) GDAL is installed with mamba/conda in the environment.

### A) With pip in a virtual environment

1. Install [GDAL](https://gdal.org/). For Ubuntu follow e.g. these [instructions](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html).
2. Create a new [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) called `gchm` by running: `python -m venv $HOME/venvs/gchm`
3. Activate the environment:`source $HOME/venvs/gchm/bin/activate`. (Check that python points to the new environment with `which python3`.)
4. Install pytorch by following the instructions on [pytorch.org](https://pytorch.org/) that match your versions. Run e.g. `python3 -m pip install torch torchvision torchaudio`
5. Install the GDAL python API matching the installed GDAL version: `python3 -m pip install GDAL==3.5.3`
6. Install all other required packages: `python3 -m pip install -r requirements.txt`
7. Install this project as an editable package called `gchm`. Make sure you are in the directory of the repository containing the file `setup.py` . 
  Run: `python3 -m pip install -e .` (Note the dot `.` at the end.)

   
### B) Mamba/conda installation

1. Install mambaforge: https://github.com/conda-forge/miniforge#mambaforge
2. Create a new environment called `gchm` with pytorch (or follow the instructions on [pytorch.org](https://pytorch.org/)):
`mamba create -n gchm python=3.10.9 pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia`
3. Activate the environment: `mamba activate gchm`. (Check that python points to the new environment: E.g. 
  `which python` should print something like: `~/mambaforge/envs/gchm/bin/python`)
4. Install gdal: `mamba install -c conda-forge gdal=3.6.2`
5. Install pytables: `mamba install -c anaconda pytables=3.7.0`
6. Install all other required packages from conda-forge using the `environment.yml` file. Change directory to the repository and run: 
`mamba env update -f environment.yml`
7. Install this project as an editable package called `gchm`. Make sure you are in the directory of the repository containing the file `setup.py` . 
  Run: `pip install -e .` (Note the dot `.` at the end.)

### Credentials for wandb
Optional. Only needed to run the training code (Not needed for deployment).
Create a file called `~/.config_wandb` containing your [weights and biases API key](https://docs.wandb.ai/quickstart): 
```
export WANDB_API_KEY=YOUR_API_KEY
```


### Credentials for AWS
Optional. This is only needed to download Sentinel-2 images from AWS on the fly using `gchm/deploy.py`. 
***Note that there are costs per GB downloaded!***

Create a file `~/.aws_configs` containing your AWS credentials as environment variables. 
```
export AWS_ACCESS_KEY_ID=PUT_YOUR_KEY_ID_HERE
export AWS_SECRET_ACCESS_KEY=PUT_YOUR_SECRET_ACCESS_KEY_HERE
export AWS_REQUEST_PAYER=requester
```
To create an AWS account go to: https://aws.amazon.com/console/
