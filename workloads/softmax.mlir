func @softmax(%input: memref<1024xf32>, %output: memref<1024xf32>) {
  %sum = constant 0.0 : f32
  affine.for %i = 0 to 1024 {
    %val = expf (load %input[%i] : memref<1024xf32>) : f32
    store %val, %output[%i] : memref<1024xf32>
    %sum = addf %sum, %val : f32
  }
  affine.for %i = 0 to 1024 {
    %val = load %output[%i] : memref<1024xf32>
    %normalized = divf %val, %sum : f32
    store %normalized, %output[%i] : memref<1024xf32>
  }
  return
}
