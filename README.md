# Seismic-Torch


This project is designed for seismic detection using machine learning techniques. It processes seismic data, specifically from lunar missions, to identify potential earthquake events.

## Installation

To set up this project, you need to install the required packages listed in the `requirements.txt` file. Follow these steps:

1. Download and extract the zip files https://wufs.wustl.edu/SpaceApps/data/space_apps_2024_seismic_detection.zip.
2. Navigate to the project directory:
   ```bash
   cd path/to/space_apps_2024_seismic_detection
   ```
3. Clone the repository or download the project files.
4. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

### Note:
- Ensure that you have Python 3.10 or later installed on your system.
- This project uses `torch`, so you may also need to install the appropriate version of CUDA if you intend to run the model on a GPU.

## Data Information

The code utilizes lunar seismic data. If you want to use Martian data instead, you will need to modify some parameters in the code. Please ensure that the data format and required features are compatible with the model.

## Directory Structure

Make sure to place the `code.py` and `requirements.txt` files in the root of the `space_apps_2024_seismic_detection` directory for the installation process to work correctly.
