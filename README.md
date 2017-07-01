# ImproveLungCancerDetection
<h2>Description</h2>
In the United States, lung cancer strikes 225,000 people every year, and accounts for $12 billion in health care costs. Early detection is critical to give patients the best chance at recovery and survival.  One year ago, the office of the U.S. Vice President spearheaded a bold new initiative, the Cancer Moonshot, to make a decade's worth of progress in cancer prevention, diagnosis, and treatment in just 5 years.  In 2017, the Data Science Bowl will be a critical milestone in support of the Cancer Moonshot by convening the data science and medical communities to develop lung cancer detection algorithms.  Using a data set of thousands of high-resolution lung scans provided by the National Cancer Institute, participants will develop algorithms that accurately determine when lesions in the lungs are cancerous. This will dramatically reduce the false positive rate that plagues the current detection technology, get patients earlier access to life-saving interventions, and give radiologists more time to spend with their patients.  This year, the Data Science Bowl will award $1 million in prizes to those who observe the right patterns, ask the right questions, and in turn, create unprecedented impact around cancer screening care and prevention. The funds for the prize purse will be provided by the Laura and John Arnold Foundation.
<h2>Dataset</h2>
<p>In this dataset, you are given over a thousand low-dose CT images from high-risk patients in DICOM format. Each image contains a series with multiple axial slices of the chest cavity. Each image has a variable number of 2D slices, which can vary based on the machine taking the scan and patient.</p>

<p>The DICOM files have a header that contains the necessary information about the patient id, as well as scan parameters such as the slice thickness.

Each patient id has an associated directory of DICOM files. The patient id is found in the DICOM header and is identical to the patient name. The exact number of images will differ from case to case, varying according in the number of slices. Images were compressed as .7z files due to the large size of the dataset.</p>
<p>
stage1.7z - contains all images for the first stage of the competition, including both the training and test set. This is file is also hosted on BitTorrent.

stage1_labels.csv - contains the cancer ground truth for the stage 1 training set images.

stage1_sample_submission.csv - shows the submission format for stage 1. You should also use this file to determine which patients belong to the leaderboard set of stage 1.

sample_images.7z - a smaller subset set of the full dataset, provided for people who wish to preview the images before downloading the large file.

data_password.txt - contains the decryption key for the image files

The DICOM standard is complex and there are a number of different tools to work with DICOM files. You may find the following resources helpful for managing the competition data:

The lite version of OsiriX is useful for viewing images on OSX.

pydicom: A package for working with images in python.

oro.dicom: A package for working with images in R.

Mango: A useful DICOM viewer for Windows users.
</p>

<h2>Sample Dataset</h2>
<p>Following is the smaple CT Scan of one of the patient. There are total of 20 slices.</p>
<img src="https://github.com/alerazabhatti/ImproveLungCancerDetection/blob/master/img/CTScan.png">
<p>Some more screenshots of the sample data are below.</p>
<img src="https://github.com/alerazabhatti/ImproveLungCancerDetection/blob/master/img/1.png">

<img src="https://github.com/alerazabhatti/ImproveLungCancerDetection/blob/master/img/2.png">

<img src="https://github.com/alerazabhatti/ImproveLungCancerDetection/blob/master/img/3.png">

<h2>Dependencies</h2>

<p>This project requires Python 3 and the following Python libraries installed:

<p>NumPy</p>
<p>SciPy</p>
<p>Scikit</p>
<p>Tflearn</p>
<p>Pandas</p>
<p>OpenCV</p>
<p>Matplotlib</p>
<p>Jupyter</p>
<p>dicom</p>
<p>h5py</p>
<p>hgf5</p>
<p>pip</p>

<h2>How to Run the Model</h2>
<p>it can be run directly by using the following command through environment terminal</p>
<p>python lung_cancer_detection_test.py</p>

<h2>Training</h2>

<p>As the dataset was very large (more than 500 GBs) so it could not fit into the main memory, so we decided to reduce it to 300 patients. So the data set was first peprocessed by using Jupyter Notebook</p>

<p>The model is trained using the following CNN</p>

<p>network = input_data(shape=[None, 20, 50, 50])</p>

<p>network = conv_2d(network, 64, 3, activation='relu')</p>

<p>network = max_pool_2d(network, 2)</p>

<p>network = conv_2d(network, 32, 3, activation='relu')</p>

<p>network = max_pool_2d(network, 2)</p>

<p>network = conv_2d(network, 32, 3, activation='relu')</p>

<p>network = max_pool_2d(network, 2)</p>

<p>network = fully_connected(network, 256, activation='relu')</p>

<p>network = dropout(network, 0.75)</p>

<p>network = fully_connected(network, 256, activation='relu')</p>

<p>network = dropout(network, 0.75)</p>

<p>network = fully_connected(network, 2, activation='softmax')</p>

<p>network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)</p>

<p>The train model is giving the acuuracy of 0.8031</p>
<img src="https://github.com/alerazabhatti/ImproveLungCancerDetection/blob/master/img/train_accuracy.png">

<h2>Testing</h2>

<p>Testing has been done on 100 patients.</p>
<p>Following is the screenshot showing the accuracy of the model.</p>
<img src="https://github.com/alerazabhatti/ImproveLungCancerDetection/blob/master/img/accuracy.png">


<h2>References</h2>
<p>https://www.kaggle.com/sentdex/first-pass-through-data-w-3d-convnet</p>




