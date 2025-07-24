# Linear Model Implementation what I understood

This guide is a walkthrough based on the onboarding requirements.

## Overview

The linear model implements a simple architecture that:
1. Averages neural data across all electrodes
2. Splits time series into bins of predetermined size
3. Uses linear regression to predict future bins from past bins using L2 loss

## Prerequisites

Before starting, ensure you have:
- Python 3.8+ with pip
- BrainTreebank dataset downloaded

## Step-by-Step Implementation

### 1. Environment Setup

First, set up Python environment:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
               \/ this was my other test had to install a new verion of pytorch because the current version wouldnt accept my gpu architecture
**GPU Support (RTX 5090/4090+ users):** If you have a newer GPU, you may need PyTorch nightly:
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 2. Data Configuration

Ensure your data paths are configured correctly:

1. **BrainTreebank data** should be in `./braintreebank/`
2. **Fix the data path** in `subject/braintreebank.py`:
   ```python
   BRAINTREEBANK_ROOT_DIR = "/path/to/your/project/braintreebank"
   ```


### 3. Implement the Linear Model

this is a kind of simple linear model. Key components:

#### Model Architecture
```python
class LinearModel(nn.Module):
    def __init__(self, input_dim, bin_size=10):
        super().__init__()
        self.bin_size = bin_size
        self.linear = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        #average across electrodes: [batch, time, channels] -> [batch, time]  
        x_avg = x.mean(dim=-1)
        
        #reshape into bins: [batch, time] -> [batch, num_bins, bin_size]
        batch_size, time_len = x_avg.shape
        num_bins = time_len // self.bin_size
        if num_bins == 0:
            return x_avg.unsqueeze(-1)
            
        x_binned = x_avg[:, :num_bins * self.bin_size].view(batch_size, num_bins, self.bin_size)
        
        #predict future bins from past bins
        predictions = self.linear(x_binned)
        return predictions
```

#### Training Configuration
```python
class YourNameLinearTraining(TrainingSetup):
    def __init__(self):
        super().__init__()
        self.input_dim = 10  #bin size
        
    def get_model(self):
        return BFModule(
            model=LinearModel(input_dim=self.input_dim),
            # ... other configurations
        )
        
    def get_loss_fn(self):
        return nn.MSELoss()  #L2 loss
```

#### Key Implementation Details


1. Simplify the forward pass:
   - Average across electrodes first
   - Use simple MSE loss between predictions and targets
   - Focus on time-series prediction

2. Feature generatio for evaluation:
   ```python
   def generate_frozen_features(self, batch):
       with torch.no_grad():
           return self.model(batch["neural"])
   ```

### 5. Run Training

Test your implementation:

```bash
#Activate environment
source .venv/bin/activate

#run training (start small for testing)
python3 pretrain.py \
    --training.setup_name your_name_linear \
    --cluster.cache_subjects 0 \
    --cluster.eval_at_beginning 0 \
    --training.n_epochs 2 \
    --training.batch_size 10
```

### 6. Monitor Training

You should see output like:
```
Linear Model with 110 parameters
Loading training split: 10 subjects found
Epoch 1/2: 100%|██| 317/317 [02:30<00:00, loss=0.0449]
Epoch 2/2: 100%|██| 317/317 [02:30<00:00, loss=0.0001]
Training completed successfully!
```

### 7. Run Evaluation

After training, evaluate your model:

```bash
python3 pretrain.py \
    --training.setup_name your_name_linear \
    --cluster.eval_only 1 \
    --cluster.eval_at_beginning 0
```

## Expected Results

- **Model size**: ~100-200 parameters (much smaller than transformer models)
- **Training time**: ~2-5 minutes per epoch on 5090
- **Performance**: Will be poor compared to transformers, but that's expected for this simple baseline

## Troubleshooting

### Common Issues

1. **CUDA Compatibility Errors**:
   - Install PyTorch nightly for newer GPUs
   - Use `CUDA_VISIBLE_DEVICES=-1` to force CPU if needed

2. **File Not Found Errors**:
   - Check data paths in `subject/braintreebank.py`
   - Ensure `corrupted_elec.json` is in the right location

3. **Import Errors**:
   - Verify all dependencies are installed
   - Check that your setup file follows the naming convention

4. **Training Setup Not Found**:
   - Filename must match `--training.setup_name` parameter
   - File must be in `training_setup/` directory

## Next Steps

Once your linear model works:

1. **Experiment with architectures**: Try different bin sizes, add layers
2. **Improve preprocessing**: Add normalization, filtering
3. **Advanced models**: Move to transformer-based architectures
4. **Evaluation**: Run neuroprobe evaluations to assess representation quality

## Further Reading

- See `training_setup/roshnipm.py` for another linear model example
- Check `quickstart.ipynb` for data exploration
- Review `evaluation/neuroprobe/` for understanding evaluation metrics
