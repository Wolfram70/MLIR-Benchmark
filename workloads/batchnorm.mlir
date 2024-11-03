func @batchnorm(%input: memref<1024xf32>, %mean: f32, %variance: f32, %gamma: f32, %beta: f32, %output: memref<1024xf32>) {
  affine.for %i = 0 to 1024 {
    %val = load %input[%i] : memref<1024xf32>
    %norm = divf (subf %val, %mean), sqrtf (addf %variance, constant 1e-5 : f32) : f32
    %scale = mulf %gamma, %norm : f32
    %shift = addf %scale, %beta : f32
    store %shift, %output[%i] : memref<1024xf32>
  }
  return
}
