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
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";
	int bin_size = 20;
	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if ((strcmp(argv[i], "-b") == 0) && (i < (argc - 1))) { bin_size = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		//Load the Image
		CImg<unsigned char> image_input;
		image_input.load(image_filename.c_str());
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

		//Get the maximum value of a pixel from the image
		int maximumPixelIntensity = image_input.max();

		//Event to track time for all operations to take place
		cl::Event profEvent;

		//Vectors to contain values from the output of the buffers
		std::vector<unsigned int> frequency_histogram(vector_elements);
		std::vector<unsigned int> cumulative_histogram(vector_elements);
		std::vector<unsigned char> normalized_histogram(vector_elements);
		std::vector<unsigned char> output_image_buffer(image_input.size());

		//Padding Calculation
		int numberToAdd = bin_size - (image_input.size() % bin_size); // Calculates the number of elements to pad by
		if (numberToAdd == bin_size) { numberToAdd = 0; } // Sets number to add to 0, if it equals the number of bins

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size()+(sizeof(unsigned char)*numberToAdd));//Padding 
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size());
		cl::Buffer histogram_buffer(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer extra_cumulative_buffer(context, CL_MEM_READ_WRITE, vector_size);
		cl::Buffer normalized_hist_buffer(context, CL_MEM_READ_WRITE, vector_size_char);
		cl::Buffer numOfBins(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer maximumValue(context, CL_MEM_READ_ONLY, sizeof(int));

		//4.1 Copy data to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0], NULL, &profEvent);
		queue.enqueueWriteBuffer(numOfBins, CL_TRUE, 0, sizeof(int), &bin_size, NULL, &profEvent);
		queue.enqueueWriteBuffer(maximumValue, CL_TRUE, 0, sizeof(int), &maximumPixelIntensity, NULL, &profEvent);
		queue.enqueueFillBuffer(histogram_buffer, 0, 0, vector_size);
		queue.enqueueFillBuffer(extra_cumulative_buffer, 0, 0, vector_size);

		//4.2 Setup the kernels (i.e. device code)
		cl::Kernel histogramKern = cl::Kernel(program, "histogramVals"); //Kernel to calculate the histogram values
		histogramKern.setArg(0, dev_image_input);
		histogramKern.setArg(1, numOfBins);
		histogramKern.setArg(2, maximumValue);
		histogramKern.setArg(3, histogram_buffer);
		histogramKern.setArg(4, cl::Local(vector_size));
		
		string kernelName = "scan_bl";
		if (!(bin_size & (bin_size - 1)) == 0) { kernelName = "scan_hs"; cerr << "Running on Hillis-Steele due to bin size\n"; }
		cl::Kernel cumulativeKern = cl::Kernel(program, kernelName.c_str()); //Kernel to calculate cumulative histogram values
		cumulativeKern.setArg(0, histogram_buffer);
		cumulativeKern.setArg(1, extra_cumulative_buffer);

		cl::Kernel normalizeKern = cl::Kernel(program, "normHistogramVals"); //Kernel to normalize histogram values
		normalizeKern.setArg(0, histogram_buffer);
		normalizeKern.setArg(1, maximumValue);
		normalizeKern.setArg(2, normalized_hist_buffer);

		cl::Kernel mapKern = cl::Kernel(program, "mapHistogram"); //Kernel to map histogram values
		mapKern.setArg(0, dev_image_input);
		mapKern.setArg(1, normalized_hist_buffer);
		mapKern.setArg(2, maximumValue);
		mapKern.setArg(3, numOfBins);
		mapKern.setArg(4, dev_image_output);

		//Run the kernels and read from the bufffers
		queue.enqueueNDRangeKernel(histogramKern, cl::NullRange, cl::NDRange(image_input.size()+numberToAdd), cl::NDRange(vector_elements), NULL, &profEvent);//Kernel for calculating the histogram values
		queue.enqueueReadBuffer(histogram_buffer, CL_TRUE, 0, vector_size, &frequency_histogram[0]);

		//Removes the extra padded values from the intensity histogram, rewrites corrected answer back to the intensity histogram buffer
		if (numberToAdd > 0) {
			frequency_histogram[0] -= numberToAdd;
			queue.enqueueWriteBuffer(histogram_buffer, CL_TRUE, 0, vector_size, &frequency_histogram[0], NULL, &profEvent);
		}

		queue.enqueueNDRangeKernel(cumulativeKern, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &profEvent);//Kernel for calculating the cumulative histogram values
		queue.enqueueReadBuffer(histogram_buffer, CL_TRUE, 0, vector_size, &cumulative_histogram[0]);
		queue.enqueueNDRangeKernel(normalizeKern, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange, NULL, &profEvent);//Kernel for normalizing the histogram
		queue.enqueueReadBuffer(normalized_hist_buffer, CL_TRUE, 0, vector_size_char, &normalized_histogram[0]);
		queue.enqueueNDRangeKernel(mapKern, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange, NULL, &profEvent);//Kernel for mapping the histogram to the image
		
		//4.3 Copy the resulting image from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_image_buffer.size(), &output_image_buffer.data()[0], NULL, &profEvent);
		//Create output image from data vector
		CImg<unsigned char> output_image(output_image_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		//Display output image
		CImgDisplay disp_output(output_image, "output");
		
		//Display the values for the histograms to the console
		cerr << "Histogram = " << frequency_histogram << std::endl;
		cerr << "Cumulative = " << cumulative_histogram << std::endl;
		cerr << "Normalized = " << normalized_histogram << std::endl << std::endl;

		//Output kernell time
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
