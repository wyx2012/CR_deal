
The tool is developed for circRNA-RBP interaction sites identification using deep hierarchical network

#Type: Package
data：
circRNA-RBP:37 datasets

# Requirements
python                    3.8.10
numpy                     1.22.4   
pandas                    2.0.3 
scikit-learn              1.3.0    
torch                     1.11.0+cu113 
torchvision               0.12.0+cu113  
# Usage
Simple operation: 
Taking the AUF1 dataset as an example, its pairwise matrix has been placed in the file. Simply run the "wTest2.py" file to obtain the results. (The results can be seen in the cTest.txt file in the AUF1 folder of the dataset folder.)

If you want to see the results of other datasets besides AUF1, you can follow the steps below，
1.Firstly, fill in the circRNA name you need to predict in pairs.py, run pairs.py, and generate the corresponding circRNA structure file in the Dataset folder.

2. Fill in the circRNA name you need to predict in wTest2.py, run wTest.py, and the results will be saved in the RESULT file of the corresponding circRNA in the dataset.

Thank you and enjoy the tool!# CR_deal
