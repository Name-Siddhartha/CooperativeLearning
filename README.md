**Usage Instructions**

**For installing multiple versions of Python into the system**
$ sudo apt install
$ sudo apt install build-essential checkinstall 
$ sudo apt install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev
$ sudo wget https://www.python.org/ftp/python/3.x.x/Python-3.x.x.tgz
$ tar xzf Python-3.x.x.tgz 
$ cd Python-3.x.x
$ sudo ./configure --enable-optimizations
$ sudo make altinstall

**For setting up different Virtual Environments for different components**
$ sudo apt-get update
$ sudo apt-get install build-essential libssl-dev libffi-dev python-dev
$ sudo apt install python3-pip
$ sudo pip3 install virtualenv 
$ virtualenv --python=python3.x <Name of Virtual Environment>
$ source <Name of Virtual Environment>/bin/activate 
$ python --version
$ deactivate #Only to deactivate when done using virtual environment

**For setting up the different modules**
$ sudo apt-get update
$ source <path to virtual environment>/bin/activate
$ python -m pip install -r requirements.txt

**For extracting and resampling audio from video**
$ ffmpeg -i vidName.mp4 -f mp3 -ab 22050 -y -vn fName.mp3	
$ ffmpeg -i fName.mp3 -acodec pcm_s16le -ac 1 -ar 22050 fName.wav

**For caffe model**
$ ./train.sh [GPU] [prototext]
E.g., $ ./train.sh 0,1,2,3 resnet_x

$ mkdir resnet_50/logs
$ mkdir resnet_50/snapshot
$./train.sh 0,1,2,3 resnet_50 resnet_50

**Testing caffe model**
$ ~/caffe/build/tools/caffe test -gpu 0 -iterations 100 -model resnet-20/trainval.prototxt -weights resnet-20/snapshot/solver_iter_64000.caffemodel 

**For paragraph-vector**
$ python -m pip install -e .
$ python train.py start --data_file_name 'example.csv' --num_epochs 100 --batch_size 32 --num_noise_words 2 --vec_dim 100 --lr 1e-3
$ python export_vectors.py start --data_file_name 'example.csv' --model_file_name 'example_model.tar'

**General extraction of all features**
$ python extract.py <Number of Videos>
