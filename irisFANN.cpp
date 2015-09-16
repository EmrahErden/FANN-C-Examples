//g++ irisFANN.cpp -lfann -lm -o irisFANN

#include <iostream>
#include "fann.h"
#include "floatfann.h"

int main()
{
	//FANN TRAINING:
    	const unsigned int num_input = 4;
    	const unsigned int num_output = 3;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 4;
	const float desired_error = (const float) 0.001;
	const unsigned int max_epochs = 200000;
	const unsigned int epochs_between_reports = 1000;

	struct fann *ann = fann_create_standard(num_layers, num_input,
	num_neurons_hidden, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_train_on_file(ann, "irisDataSet.data", max_epochs,
	epochs_between_reports, desired_error);

	fann_save(ann, "irisFANN_float.net");

	fann_destroy(ann);

	//FANN TESTING
	fann_type *calc_out;
	fann_type input[4];

	struct fann *annR = fann_create_from_file("irisFANN_float.net");

	input[0] = 5.1;
	input[1] = 3.5;
	input[2] = 1.4;
	input[3] = 0.2;

	calc_out = fann_run(annR, input);

	std::cout << "Testing irisFANN, with data: " << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << ", "
	<< ", output: " << calc_out[0] << ", " << calc_out[1] << ", " << calc_out[2] << "\n"; 

	fann_destroy(annR);
	return 0;

}
