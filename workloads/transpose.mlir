func @transpose(%input: memref<1024x1024xf32>, %output: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %val = load %input[%i, %j] : memref<1024x1024xf32>
      store %val, %output[%j, %i] : memref<1024x1024xf32>
    }
  }
  return
}
