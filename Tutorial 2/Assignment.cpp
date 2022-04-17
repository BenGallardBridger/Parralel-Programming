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
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char** argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	string image_filename = "test.pgm";

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if ((strcmp(argv[i], "-f") == 0) && (i < (argc - 1))) { image_filename = argv[++i]; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0; }
	}

	cimg::exception_mode(0);

	//detect any potential exceptions
	try {
		CImg<unsigned short> image_input;
		image_input.load(image_filename.c_str());

		CImgDisplay disp_input(image_input, "input");

		//Part 3 - host operations
		//3.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);
		
		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;
		
		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

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
		int max = ((int)image_input.max()) + 2;
		cerr << max << "\n";
		size_t vector_elements = 128;//number of elements
		size_t vector_size = 128 * sizeof(unsigned int);//size in bytes

		//host - output
		std::vector<unsigned int> C(vector_elements);

		cl::Buffer buffer_C(context, CL_MEM_READ_WRITE, vector_size);

		//host - output
		std::vector<unsigned int> D(vector_elements);

		cl::Buffer buffer_D(context, CL_MEM_READ_WRITE, vector_size);

		size_t vector_size1 = max * sizeof(unsigned short);//size in bytes

		std::vector<unsigned short> E(vector_elements);

		cl::Buffer buffer_E(context, CL_MEM_READ_WRITE, vector_size1);

		//device - buffers
		cl::Buffer dev_image_input(context, CL_MEM_READ_ONLY, image_input.size());
		cl::Buffer dev_image_output(context, CL_MEM_READ_WRITE, image_input.size()); //should be the same as input image

		//4.1 Copy images to device memory
		queue.enqueueWriteBuffer(dev_image_input, CL_TRUE, 0, image_input.size(), &image_input.data()[0]);


		//4.2 Setup and execute the kernel (i.e. device code)
		cl::Kernel kernel = cl::Kernel(program, "histogramVals");
		kernel.setArg(0, dev_image_input);
		kernel.setArg(1, buffer_C);
		
		cl::Kernel kernelB = cl::Kernel(program, "cumHistogramVals");
		kernelB.setArg(0, buffer_C);
		kernelB.setArg(1, buffer_D);

		cl::Kernel kernelC = cl::Kernel(program, "normHistogramVals");
		kernelC.setArg(0, buffer_D);
		kernelC.setArg(1, buffer_E);

		cl::Kernel kernelD = cl::Kernel(program, "mapHistogram");
		kernelD.setArg(0, dev_image_input);
		kernelD.setArg(1, buffer_E);
		kernelD.setArg(2, dev_image_output);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		queue.enqueueNDRangeKernel(kernelB, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);
		queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, vector_size, &C[0]);
		cerr << "C = " << C;
		queue.enqueueNDRangeKernel(kernelC, cl::NullRange, cl::NDRange(vector_elements), cl::NullRange);
		queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, vector_size, &D[0]);
		cerr << "\nD = " << D;
		queue.enqueueNDRangeKernel(kernelD, cl::NullRange, cl::NDRange(image_input.size()), cl::NullRange);
		cerr << "ENQUE";
		queue.enqueueReadBuffer(buffer_E, CL_TRUE, 0, vector_size1, &E[0]);
		cerr << "\nE = " << E;
		
		vector<unsigned short> output_buffer(image_input.size());
		//4.3 Copy the result from device to host
		queue.enqueueReadBuffer(dev_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		CImg<unsigned short> output_image(output_buffer.data(), image_input.width(), image_input.height(), image_input.depth(), image_input.spectrum());
		CImgDisplay disp_output(output_image, "output");
		

		while (!disp_output.is_closed() && !disp_input.is_closed()
			&& !disp_output.is_keyESC() && !disp_input.is_closed()) {
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
