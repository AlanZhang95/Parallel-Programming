#include <cstdio>
#include <cstdlib>

#include "helper.hpp"

#define TILE_SIZE 32

__global__ void kernel(int *A0, int *Anext, int nx, int ny, int nz) {

  // INSERT KERNEL CODE HERE
  #define _Anext(xi, yi, zi) Anext[(zi) * (ny * nx) + (yi) * nx + (xi)]
  #define _A0(xi, yi, zi)    A0[(zi) * (ny * nx) + (yi) * nx + (xi)]
//  __shared__ int tileXY[TILE_SIZE][TILE_SIZE];
//  __shared__ int tileXYPrev[TILE_SIZE][TILE_SIZE];
//  __shared__ int tileXYNext[TILE_SIZE][TILE_SIZE];
  int tx = threadIdx.x;  //32
  int ty = threadIdx.y;  //32
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int row = by * TILE_SIZE + ty;
  int col = bx * TILE_SIZE + tx;
//  int depth = bz * TILE_SIZE;

    for(int i = 1; i < (nz - 1); i++){
  //    tileXYPrev[tx][ty] = _A0(tx,ty,i-1);
  //    tileXY[tx][ty] = _A0(tx,ty,i);
  //    tileXYNext[tx][ty] = _A0(tx,ty,i+1);
      if(col >= 1 && col < (nx-1) && row >= 1 && row < (ny-1)){
      Anext[(i) * (ny * nx) + (row) * nx + (col)] = //tileXYPrev[tx][ty] +
                                                  //tileXYNext[tx][ty] +
                                                  //tileXY[tx + 1][ty] +
                                                  //tileXY[tx - 1][ty] +
                                                  //tileXY[tx][ty + 1] +
                                                  //tileXY[tx][ty - 1] -
                                                  //6 * tileXY[tx][ty];

                                                  _A0(col,row,i - 1) +  //
                                                  _A0(col,row,i + 1) +
                                                  _A0(col,row + 1,i) +
                                                  _A0(col,row - 1,i) +
                                                  _A0(col + 1,row,i) +
                                                  _A0(col - 1,row,i) -
                                                  6 * _A0(col,row,i);

        }
  }

  #undef _Anext
  #undef _A0
}

void launchStencil(int* A0, int* Anext, int nx, int ny, int nz) {

  // INSERT CODE HERE

  dim3 blockDim(32,32,1);
  dim3 gridDim(ceil(nx/32.0),ceil(ny/32.0),1);
  kernel<<<gridDim, blockDim>>>(A0, Anext, nx, ny, nz);


}


static int eval(const int nx, const int ny, const int nz) {

  // Generate model
  const auto conf_info = std::string("stencil[") + std::to_string(nx) + "," +
                                                   std::to_string(ny) + "," +
                                                   std::to_string(nz) + "]";
  INFO("Running "  << conf_info);

  // generate input data
  timer_start("Generating test data");
  std::vector<int> hostA0(nx * ny * nz);
  generate_data(hostA0.data(), nx, ny, nz);
  std::vector<int> hostAnext(nx * ny * nz);

  timer_start("Allocating GPU memory.");
  int *deviceA0 = nullptr, *deviceAnext = nullptr;
  CUDA_RUNTIME(cudaMalloc((void **)&deviceA0, nx * ny * nz * sizeof(int)));
  CUDA_RUNTIME(cudaMalloc((void **)&deviceAnext, nx * ny * nz * sizeof(int)));
  timer_stop();

  timer_start("Copying inputs to the GPU.");
  CUDA_RUNTIME(cudaMemcpy(deviceA0, hostA0.data(), nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  //////////////////////////////////////////
  // GPU Gather Computation
  //////////////////////////////////////////
  timer_start("Performing GPU convlayer");
  launchStencil(deviceA0, deviceAnext, nx, ny, nz);
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  timer_start("Copying output to the CPU");
  CUDA_RUNTIME(cudaMemcpy(hostAnext.data(), deviceAnext, nx * ny * nz * sizeof(int), cudaMemcpyDefault));
  CUDA_RUNTIME(cudaDeviceSynchronize());
  timer_stop();

  // verify with provided implementation
  timer_start("Verifying results");
  verify(hostAnext.data(), hostA0.data(), nx, ny, nz);
  timer_stop();

  CUDA_RUNTIME(cudaFree(deviceA0));
  CUDA_RUNTIME(cudaFree(deviceAnext));

  return 0;
}



TEST_CASE("Stencil", "[stencil]") {

  SECTION("[dims:32,32,32]") {
    eval(32,32,32);
  }
  SECTION("[dims:30,30,30]") {
    eval(30,30,30);
  }
  SECTION("[dims:29,29,29]") {
    eval(29,29,29);
  }
  SECTION("[dims:31,31,31]") {
    eval(31,31,31);
  }
  SECTION("[dims:29,29,2]") {
    eval(29,29,29);
  }
  SECTION("[dims:1,1,2]") {
    eval(1,1,2);
  }
  SECTION("[dims:512,512,64]") {
    eval(512,512,64);
  }

}
