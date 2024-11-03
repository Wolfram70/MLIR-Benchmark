func @dotprod(%A: memref<100000xf32>, %B: memref<100000xf32>, %output: memref<1xf32>) {
  %sum = constant 0.0 : f32
  affine.for %i = 0 to 100000 {
    %a = load %A[%i] : memref<100000xf32>
    %b = load %B[%i] : memref<100000xf32>
    %prod = mulf %a, %b : f32
    %sum = addf %sum, %prod : f32
  }
  store %sum, %output[0] : memref<1xf32>
  return
}
