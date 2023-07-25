# Nvidia Docker version: 11.3.1-cudnn8-devel-ubuntu18.04

# install dependencies
apt-get install git cmake automake autoconf build-essential gawk wget

# install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
./Anaconda3-2022.10-Linux-x86_64.sh

# git clone fairseq
git clone https://github.com/facebookresearch/fairseq.git
rm -r fairseq/.git*

# fairseq_medp/
cat medp_scripts/configs.py > fairseq/fairseq/dataclass/configs.py
cat medp_scripts/train.py > fairseq/fairseq_cli/train.py
cat medp_scripts/trainer.py > fairseq/fairseq/trainer.py
cat medp_scripts/fairseq_task.py > fairseq/fairseq/tasks/fairseq_task.py
cat medp_scripts/label_smoothed_cross_entropy.py > fairseq/fairseq/criterions/label_smoothed_cross_entropy.py
cat medp_scripts/1_run_iwslt14de2en_segment_level_training.sh > fairseq/1_run_iwslt14de2en_segment_level_training.sh
cat medp_scripts/2_run_iwslt14de2en_decode.sh > fairseq/2_run_iwslt14de2en_decode.sh

# install fairseq
cd fairseq
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --editable ./
pip install Levenshtein sacremoses

# prepare data
cd fairseq/examples/translation
./prepare-iwslt14.sh

# run script
./../../
./1_run_iwslt14de2en_segment_level_training.sh &> log001_segment_level_training