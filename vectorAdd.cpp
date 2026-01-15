// Simple vectorAdd implementation for CUDA validation
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Print info about GPU
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    
    if (error_id != cudaSuccess) {
        std::cout << "CUDA Driver/Runtime Error: " << cudaGetErrorString(error_id) << std::endl;
        return 1;
    }
    
    if (deviceCount == 0) {
        std::cout << "No CUDA capable devices detected" << std::endl;
        return 1;
    }
    
    std::cout << "CUDA device count: " << deviceCount << std::endl;
    
    // Simple vector addition test
    const int numElements = 50000;
    size_t size = numElements * sizeof(float);
    
    // Allocate host memory
    std::vector<float> h_A(numElements, 1.0f);
    std::vector<float> h_B(numElements, 2.0f);
    std::vector<float> h_C(numElements, 0.0f);
    
    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    
    // Copy result back
    cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            std::cout << "Result verification failed at element " << i << std::endl;
            return 1;
        }
    }
    
    std::cout << "CUDA vectorAdd test PASSED" << std::endl;
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
