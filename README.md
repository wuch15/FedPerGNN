# FedGNN 

1. Environment Requirements
* Ubuntu 16.04
* Anaconda with Python 3.6.9
* CUDA 10.0

Note: The specific python package list of our environment is included in the requirements.txt. The tensorflow version can be 1.12~1.15.
The installation may need several minutes if there is no environmental conflicts.

2. Hardware requirements
Needs a Linux server with larger than 32GB memory. GPU cores are optional but recommended.

3. Running

* Download datasets from their original sources
* Convert then into matlab matrix formats (rows: users, columns: items, value: ratings)
* Execute "python run.py"

Note: The logs will show the training loss and the test results. The estimated running time is from tens of minutes to hours, depending on the dataset size.
