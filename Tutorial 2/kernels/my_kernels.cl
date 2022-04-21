//converts values into histogram
kernel void histogramVals(global const uchar* A, global const int* binSize, global const int* maximum, global uint* B) {
	int id = get_global_id(0);
	int bin_num = (A[id]/(float)maximum[0])*binSize[0]-1;

	atomic_inc(&B[bin_num]);
}

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
kernel void scan_hs(global uint* A, global uint* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	global uint* C;

	for (int stride = 1; stride < N; stride *= 2) {
		B[id] = A[id];
		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}

kernel void cumHistogramVals(global const uint* A, global uint* B) {
	int id = get_global_id(0);
	int size = get_global_size(0);
	int temp = 0;
	for (int i = 0; i < id + 1; i++) {
		temp += A[i];
	}
	B[id] = temp;
}

kernel void normHistogramVals(global const uint* A, global const int* maximum, global int* B) {
	int id = get_global_id(0);
	int size = get_global_size(0);
	int max = A[size - 1];
	int temp = (A[id] / (float)max) * maximum[0];
	if (temp > maximum[0]) {

		B[id] = (int)(maximum[0]);
	}
	else {
		B[id] = (int)temp;
	}
}

kernel void mapHistogram(global const uchar* A, global const int* B, global const int* maximum, global const int* binSize, global uchar* C) {
	int id = get_global_id(0);
	int binNum = (A[id] / (float)maximum[0]) * binSize[0];
	C[id] = B[binNum];
}
