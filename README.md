# Speaker-Recognition
Automatic Speaker Recognition algorithms in Python

This repository contains Python programs that can be used for Automatic Speaker Recognition. ASR is done by extracting MFCCs and LPCs from each speaker and then forming a speaker-specific codebook
of the same by using Vector Quantization (I like to think of it as a fancy name for NN-clustering). 
After that, the system is trained and tested for 8 different speakers. 

Create virtualenv with:

	virtualenv -p python3 .env
	. .env/bin/activate
	pip install -r requirements.txt

To test the algorithm, run test.py. Certain parameters are open to be changed, such as the order of LPC coefficients, the number of Mel filterbanks and the number of centroids in each codebook.
Everything is included in the repository, including .wav files for testing and training, hence cloning it and running test.py should work. 

A PDF has been included that explains the theory and provides links to relevant websites and projects.
