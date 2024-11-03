func @reduce_sum(%input: memref<1024x1024xf32>, %output: memref<1xf32>) {
  %sum = constant 0.0 : f32
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %val = load %input[%i, %j] : memref<1024x1024xf32>
      %sum = addf %sum, %val : f32
    }
  }
  store %sum, %output[0] : memref<1xf32>
  return
}
