//converts values into histogram
//kernel void histogramVals(global const uchar* A, global const int* binSize, global const int* maximum, global uint* B) {
//	int id = get_global_id(0);
//	int bin_num = (A[id]/(float)maximum[0])*(binSize[0]-1);
//
//	atomic_inc(&B[bin_num]);
//}

//histogrammer
kernel void histogramVals(global const uchar* A, global const int* binSize, global const int* maximum, global uint* B, local uint* localH) {
	int id = get_global_id(0);
	int bin_num = (A[id] / (float)maximum[0]) * binSize[0]-1;
	barrier(CLK_LOCAL_MEM_FENCE);
	atomic_inc(&localH[bin_num]);
	barrier(CLK_LOCAL_MEM_FENCE);
	if (id < binSize[0]) {
		if (localH[id] >0)
		{
			atomic_add(&B[id], localH[id]);
		}
	}
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

//Blelloch style scan with extra buffer to allow for edits to make inclusive
kernel void scan_bl(global uint* A, global uint* localBuff) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;
	
	localBuff[id] = A[id];
	barrier(CLK_GLOBAL_MEM_FENCE);

	// Up-sweep
	for (int stride = 1; stride < N; stride *= 2) {
		if (((id + 1) % (stride * 2)) == 0)
			localBuff[id] += localBuff[id - stride];
		barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}

	// Down-sweep
	if (id == 0) localBuff[N - 1] = 0;	 // Exclusive scan
	barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step

	for (int stride = N / 2; stride > 0; stride /= 2) {
		if (((id + 1) % (stride * 2)) == 0) {
			t = localBuff[id];
			localBuff[id] += localBuff[id - stride];  // Reduce 
			localBuff[id - stride] = t;	// Move
		}
		barrier(CLK_GLOBAL_MEM_FENCE); // Sync the step
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
	if (id > 0 && id < (N)) {
		A[id - 1] = localBuff[id];
	}
	if (id == (N - 1)) {
		A[id] = localBuff[N - 1] + A[N - 1];
	}

}
//Basic Naive implementation of cumulative histogram scan
kernel void cumHistogramVals(global const uint* A, global uint* B) {
	int id = get_global_id(0);
	int size = get_global_size(0);
	int temp = 0;
	for (int i = 0; i < id + 1; i++) {
		temp += A[i];
	}
	B[id] = temp;
}
//Normalize histogram values
kernel void normHistogramVals(global const uint* A, global const int* maximum, global uchar* B) {
	int id = get_global_id(0);
	int size = get_global_size(0);
	int max = A[size-1];
	int temp = (A[id] / (float)max) * maximum[0];
	B[id] = temp;
}
//Map histogram values
kernel void mapHistogram(global const uchar* A, global const uchar* B, global const int* maximum, global const int* binSize, global uchar* C) {
	int id = get_global_id(0);
	int binNum = (A[id] / (float)maximum[0]) * (binSize[0]-1);
	C[id] = B[binNum];
}
