1. Libraries required to run the project

numpy==2.1.3
opencv-python==4.10.0.84

(Libraries can be found in requirements.txt as well)

2. How to run each task and where to look for the output file.

All tasks are implemented within the same python script "tasks.py".

In the generate_output() function, the paths of the input/output folders are defined.
Valid configuration for the fake test and the evaluation test are provided in the function:
input_folder = "evaluare/fake_test"
output_folder = "evaluare/fisiere_solutie/464_Andrei_Timotei/"

To run the script and generate the output for all tasks, run the file as usual: "python tasks.py".

Note: The script "tasks.py" should be at the same directory level as the "antrenare", "evaluare" and "imagini_auxiliare", but also "classifier.py", "templates.py" and "utils.py".
