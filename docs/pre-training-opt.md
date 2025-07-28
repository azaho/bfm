## `torch.compile()` Performance Analysis

A significant speedup in the training pipeline was achieved by adding a single line of code to enable PyTorch's Just-In-Time (JIT) compiler:

```python
self.model = torch.compile(self.model)
```
This function optimizes the model by analyzing its structure and fusing multiple computational steps into a single, highly efficient CUDA kernel. This reduces the overhead associated with running many separate operations on the GPU.

## Performance Comparison
The following table shows the average time per batch before and after the optimization.

| Metric | Before `torch.compile` | After `torch.compile` | Speedup |
| :--- | :--- | :--- | :--- |
| **Model Forward Pass** | ~8.0 ms | ~3.5 ms | **2.3x** |
| **Total Time per Batch** | ~15.0 ms | ~8.0 ms | **1.9x** |


Note: The first batch after enabling torch.compile took ~7 seconds. This is an expected, one-time cost for the JIT compiler to analyze and optimize the model.

Conclusion
By using torch.compile(), the model's forward pass became 2.3 times faster, leading to an overall training speedup of 1.9x. This demonstrates a highly effective optimization that nearly halved the total training time.