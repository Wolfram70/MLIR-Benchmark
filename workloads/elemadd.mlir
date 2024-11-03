func @elemadd(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      %a = load %A[%i, %j] : memref<1024x1024xf32>
      %b = load %B[%i, %j] : memref<1024x1024xf32>
      %sum = addf %a, %b : f32
      store %sum, %C[%i, %j] : memref<1024x1024xf32>
    }
  }
  return
}
