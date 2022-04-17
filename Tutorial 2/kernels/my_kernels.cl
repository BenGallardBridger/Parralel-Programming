//converts values into histogram
kernel void histogramVals(global const ushort* A, global uint* B) {
	int id = get_global_id(0);
	int bin_num = (A[id]/65537.f)*127;
	atomic_inc(&B[bin_num]);
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

kernel void normHistogramVals(global const uint* A, global ushort* B) {
	int id = get_global_id(0);
	int size = get_global_size(0);
	int max = A[size - 1];
	int temp = ((double)A[id] / (double)max) * (double)size;
	if (temp > size - 1) {

		B[id] = (ushort)(size - 1);
	}
	else {
		B[id] = (ushort)temp;
	}
}

kernel void mapHistogram(global const ushort* A, global const ushort* B, global ushort* C) {
	int id = get_global_id(0);
	C[id] = B[A[id]];
}
