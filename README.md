# Physics-Informed Neural Networks for Periodic Hill

# Project Description
This project develops a PINN model to simulate fluid flow in simple geometries, such as periodic hills. The project integrates a neural network that incorporates the Navier-Stokes equations, which govern fluid flow. The model is trained and validated using available datasets or simplified scenarios.

# File Structure

 PINN_periodicHill
 ┣ data                                                 # Contains input .npy files
 ┃ ┣ komegasst_PHLL_case_1p0_Cx.npy
 ┃ ┣ komegasst_PHLL_case_1p0_Cy.npy
 ┃ ┣ komegasst_PHLL_case_1p0_Ux.npy
 ┃ ┣ komegasst_PHLL_case_1p0_Uy.npy
 ┃ ┣ komegasst_PHLL_case_1p0_p.npy
 ┃ ┗ komegasst_PHLL_case_1p0_gradU.npy
 ┣  npy-matlab                                         # MATLAB library for reading .npy files
 ┃ ┣ constructNPYheader.m
 ┃ ┣ datToNPY.m
 ┃ ┣ readNPY.m
 ┃ ┣ readNPYheader.m
 ┃ ┗ writeNPY.m
 ┣ Periodic_Hill_PINN.m                               # Main MATLAB script
 ┣ README.md                                          # Instructions to run the project
 ┗ LICENSE                                            # License file


# Dependencies & Requirements

Before running the project, ensure you have:
-MATLAB R2022a or later
-Deep Learning Toolbox™
-npy-matlab Library (included in this repository for reading .npy files)

# Installation & Running the Code

Follow these steps to set up and run the project:

# STEP-1:Clone the Repository to your local machine

1.1:Open a terminal or Git Bash :
1.2: Navigate to the directory where you want to clone the repository
	eg: cd "C:\Users\Razzaq\Documents"
1.3: Clone the repository using
	git clone https://github.com/razzaqmohammed22/PINN_periodicHill.git
1.4: After cloning, navigate to the project directory:
    	cd PINN_periodicHill


# STEP-2:Open MATLAB and Set Up the working directory

2.1: Launch MATLAB.
2.2: In the MATLAB Current Folder window, navigate to the cloned folder
	eg:  C:\Users\Razzaq\Documents\PINN_periodicHill

OR directly open the .m file from the project directory if you have matlab installed in your machine 

# Run the Main Script
Execute the MATLAB script to train and test the PINN model:

Periodic_Hill_PINN


# Expected Output

After running the main script 
Epoch number , Training Loss = , Validation Loss = 
PINN Training Time:
Comparison Table (10 Validation Samples):
Full Dataset Mean Absolute Error: 
Validation Mean Absolute Error: 
Full Dataset Root Mean Squared Error: 
Validation Mean Absolute Error: 
Analysis Summary:
3 figures and a plot

# Troubleshooting & FAQs

 File Not Found Error?
Ensure you are in the correct directory and that the  .npy  files are present.



# License
This project is licensed under the MIT License. See the `LICENSE` file for details.
