func.func @matmul(%A: memref<1024x1024xf32>, %B: memref<1024x1024xf32>, %C: memref<1024x1024xf32>) {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %a = memref.load %A[%i, %k] : memref<1024x1024xf32>
        %b = memref.load %B[%k, %j] : memref<1024x1024xf32>
        %c = memref.load %C[%i, %j] : memref<1024x1024xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        memref.store %sum, %C[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}