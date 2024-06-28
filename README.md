# Repository for PhD of Jan

## Setup

1. Install **git** as a VCS (version control system): [https://git-scm.com/download](https://git-scm.com/download)
2. Install **Microsoft Visual Studio Code** as an IDE (integrated development environment): [https://code.visualstudio.com/download#](https://code.visualstudio.com/download#)
3. Install **miniconda** to manage the virtual environments. Open a terminal and paste the following lines of code:
  ```
  mkdir -p ~/workspace/miniconda3
  curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/workspace/miniconda3/miniconda.sh
  bash ~/workspace/miniconda3/miniconda.sh -b -u -p ~/workspace/miniconda3
  rm -rf ~/workspace/miniconda3/miniconda.sh
  ```
4. Init miniconda to use in the terminal:
  ```
  ~/workspace/miniconda3/bin/conda init bash
  ~/workspace/miniconda3/bin/conda init zsh
  ```
5. Download the embeddings as a .zip file from the Drive, unzip them and move them into the "embeddings" folder: [Link](https://drive.google.com/file/d/1AxxrxIE8KR9C1P21w_mDgRp2DA0llbD1/view?usp=drive_link)
6. Create a new environment with the following terminal command:
  ```
  conda env create --file environment.yml
  ```
7. Activate the environment with the following terminal command:
  ```
  conda activate jan-phd
  ```
8. Adapt the parameters in the `pipeline.ipynb` notebook and run the cells one by one. 