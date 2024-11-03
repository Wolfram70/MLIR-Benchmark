func @conv2d(%input: memref<256x256xf32>, %kernel: memref<3x3xf32>, %output: memref<254x254xf32>) {
  affine.for %i = 0 to 254 {
    affine.for %j = 0 to 254 {
      %sum = constant 0.0 : f32
      affine.for %ki = 0 to 3 {
        affine.for %kj = 0 to 3 {
          %in = load %input[%i + %ki, %j + %kj] : memref<256x256xf32>
          %ker = load %kernel[%ki, %kj] : memref<3x3xf32>
          %prod = mulf %in, %ker : f32
          %sum = addf %sum, %prod : f32
        }
      }
      store %sum, %output[%i, %j] : memref<254x254xf32>
    }
  }
  return
}
