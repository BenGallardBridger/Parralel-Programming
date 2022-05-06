/*
Assignment 1
Parallel Programming CMP3752M
Ben Gallard-Bridger 19693078

For this assignment I developed this application to calculate a pixel intensity histogram,
normalize the pixel intensity histogram of a variety of images, then back project this histogram onto the original image.
The images (.ppm and .pgm format) which are supported are: 
	1.	Greyscale
	2.	RGB
	3.	8-bit
	4.	16-bit (Converts into an 8-bit image) / using: https://github.com/dtschump/CImg/issues/218
	5.	Small and Large
	6.	Combination of all the previous. 

Some optimization efforts have been made for the creation of the histogram, using local memory to store local versions of the histogram,
so less atomic functions are needed for the global version of the histogram. There are 3 available scans to allow for the cumulative histogram:
	1.	Simple Scan – Uses Atomic Functions on Global Memory – Very Slow
	2.	Hillis-Steele – Inclusive Scan
	3.	Blelloch – Edited to allow for inclusive scan – Switches to Hillis-Steele when bin size is not a power of 2.

Extra features which have been developed are:
	1.	Variable bin size
	2.	Padding to allow for any bin numbers

Variables can be set/changed when running through the CMD.
To change the bin size, use identifier ‘-b’ followed by a space and the number of bins you would like.
To change the file, use identifier ‘-f’ followed by a space and the file name you would like.
To change the scan, use identifier ‘-s’ followed by a space and the characters for the scan.
You can also change the platform and device using the identifiers ‘-p’ / ‘-d’ respectively, along with the id of the platform/device.
To finalize, the times taken for all the kernels to run are output to the console.

*/


#include <iostream>
#include <vector>

#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -f : input image file (default: test.pgm)" << std::endl;
	std::cerr << "  -b : bin size (default: 128)" << std::endl;
	std::cerr << "  -s : scan (default: Blelloch - Goes to Hillis-Steele if bin size if not a power of 2)(options: bl-Blelloch/hs-Hillis-Steele/si-Simple)" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	int bin_size = 32;

	string scanName = "hs";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-s") == 0) && (i < (argc - 1))) { scanName = argv[++i]; }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { bin_size = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		//Load the Image
		CImg<unsigned short> img0(image_filename.c_str());
		//Convert image to 8 bit
		CImg<unsigned char> image_input = (img0 / (img0.max() > 255 ? 257 : 1));
		//Display the image
		CImgDisplay disp_input(image_input, "input");

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		
		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		
		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//3.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		//Part 4 - device operations
		size_t vector_elements = bin_size;//number of elements
		size_t vector_size = bin_size * sizeof(unsigned int);//size in bytes
		size_t vector_size_char = bin_size * sizeof(unsigned char);//size in bytes
		size_t picture_size = image_input.size() * sizeof(unsigned char); //size of picture in bytes
		size_t single_int_size = sizeof(int);
		//Get the maximum value of a pixel from the image
		int maximumPixelIntensity = image_input.max();

		//Event to track time for all operations to take place
		cl::Event profEvent;

		//Vectors to contain values from the output of the buffers
		std::vector<unsigned int> frequency_histogram(vector_elements);
		std::vector<unsigned int> frequency_histogram1(vector_elements);
		std::vector<unsigned int> frequency_histogram2(vector_elements);
		std::vector<unsigned char> output_image_buffer(image_input.size());

		//Padding Calculation
		int numberToAdd = bin_size - (image_input.size() % bin_size); // Calculates the number of elements to pad by
		if (numberToAdd == bin_size) { numberToAdd = 0; } // Sets number to add to 0, if it equals the number of bins
		size_t padded_size = picture_size + (numberToAdd * sizeof(unsigned char));

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, padded_size);//Padding 
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, picture_size);
		cl::Buffer histogram_buffer(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer cumulative_buffer(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer normalized_hist_buffer(context, CL_MEM_READ_WRITE, vector_size_char);
		cl::Buffer numOfBins(context, CL_MEM_READ_ONLY, single_int_size);
		cl::Buffer maximumValue(context, CL_MEM_READ_ONLY, single_int_size);

		//4.1 Copy data to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, picture_size, &image_input.data()[0], NULL, &profEvent);
		queue.enqueueWriteBuffer(numOfBins, CL_TRUE, 0, single_int_size, &bin_size, NULL, &profEvent);
		queue.enqueueWriteBuffer(maximumValue, CL_TRUE, 0, single_int_size, &maximumPixelIntensity, NULL, &profEvent);

		//4.2 Setup the kernels (i.e. device code)
		cl::Kernel histogramKern = cl::Kernel(program, "histogramVals"); //Kernel to calculate the histogram values
		histogramKern.setArg(0, dev_image_input);
		histogramKern.setArg(1, numOfBins);
		histogramKern.setArg(2, maximumValue);
		histogramKern.setArg(3, histogram_buffer);
		histogramKern.setArg(4, cl::Local(vector_size));
		
		//Set the name of the scan kernel to use to allow for the user to choose what scan to enact
		string scanKernel = "scan_bl";
		if (scanName == "hs") {
			scanKernel = "scan_hs";
		}
		else if (scanName == "si") {
			scanKernel = "simpleScan";
		}
		if (!(bin_size & (bin_size - 1)) == 0 && scanKernel == "scan_bl") { scanKernel = "scan_hs"; cerr << "Running on Hillis-Steele due to bin size\n"; }

		cl::Kernel cumulativeKern = cl::Kernel(program, scanKernel.c_str()); //Kernel to calculate cumulative histogram values
		cumulativeKern.setArg(0, histogram_buffer);
		cumulativeKern.setArg(1, cumulative_buffer);

		cl::Kernel normalizeKern = cl::Kernel(program, "normHistogramVals"); //Kernel to normalize histogram values
		if (scanKernel == "scan_bl") {
			normalizeKern.setArg(0, histogram_buffer);
		}
		else {
			normalizeKern.setArg(0, cumulative_buffer);
		}
		normalizeKern.setArg(1, maximumValue);
		normalizeKern.setArg(2, normalized_hist_buffer);

		cl::Kernel mapKern = cl::Kernel(program, "mapHistogram"); //Kernel to map histogram values
		mapKern.setArg(0, dev_image_input);
		mapKern.setArg(1, normalized_hist_buffer);
		mapKern.setArg(2, maximumValue);
		mapKern.setArg(3, numOfBins);
		mapKern.setArg(4, dev_image_output);


		//Run the kernels and read from the bufffers
		//Kernel for calculating the histogram values
		queue.enqueueNDRangeKernel(histogramKern, cl::NullRange, cl::NDRange(image_input.size()+numberToAdd), cl::NDRange(vector_elements), NULL, &profEvent);
		queue.enqueueReadBuffer(histogram_buffer, CL_TRUE, 0, vector_size, &frequency_histogram[0]);

		//Removes the extra padded values from the intensity histogram, rewrites corrected answer back to the intensity histogram buffer
		if (numberToAdd > 0) {
			frequency_histogram[0] -= numberToAdd;
			queue.enqueueWriteBuffer(histogram_buffer, CL_TRUE, 0, vector_size, &frequency_histogram[0], NULL, &profEvent);
		}
		cerr << frequency_histogram << endl;
		//Kernel for calculating the cumulative histogram values
		queue.enqueueNDRangeKernel(cumulativeKern, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &profEvent);
		queue.enqueueReadBuffer(cumulative_buffer, CL_TRUE, 0, vector_size, &frequency_histogram1[0]);
		cerr << frequency_histogram1 << endl;
		//Kernel for normalizing the histogram
		queue.enqueueNDRangeKernel(normalizeKern, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &profEvent);
		//
		queue.enqueueReadBuffer(normalized_hist_buffer, CL_TRUE, 0, vector_size, &frequency_histogram2[0]);
		cerr << frequency_histogram2 << endl;
		//Kernel for mapping the histogram to the image
		queue.enqueueNDRangeKernel(mapKern, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &profEvent);
		
		//4.3 Copy the resulting image from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, picture_size, &output_image_buffer.data()[0], NULL, &profEvent);
		//Create output image from data vector
		CImg<unsigned char> output_image(output_image_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//Display output image
		CImgDisplay disp_output(output_image, "output");

		//Output kernel time
		std::cout << "\nKernel execution time [ns]:" <<
			profEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
			profEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetFullProfilingInfo(profEvent, ProfilingResolution::PROF_US)
			<< std::endl;

		//Tells the application to wait until both images are closed
		while (!disp_input.is_closed() && !disp_output.is_closed()
			&& !disp_input.is_keyESC() && !disp_output.is_keyESC()) {
			disp_input.wait(1);
			disp_output.wait(1);
		}

	}
	catch (const cl::Error& err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (CImgException& err) {
		std::cerr << "ERROR: " << err._message;
	}

	return 0;
}
