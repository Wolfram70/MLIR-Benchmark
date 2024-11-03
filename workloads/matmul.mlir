func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %a = load %A[%i, %k] : memref<1024x1024xf32>
        %b = load %B[%k, %j] : memref<1024x1024xf32>
        %c = load %C[%i, %j] : memref<1024x1024xf32>
        %prod = mulf %a, %b : f32
        %sum = addf %c, %prod : f32
        store %sum, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}