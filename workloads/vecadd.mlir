func @vecadd(%A: memref<100000xf32>, %B: memref<100000xf32>, %C: memref<100000xf32>) {
  affine.for %i = 0 to 100000 {
    %a = load %A[%i] : memref<100000xf32>
    %b = load %B[%i] : memref<100000xf32>
    %sum = addf %a, %b : f32
    store %sum, %C[%i] : memref<100000xf32>
  }
  return
}
