# UDIG: Uniform Discretized Integrated Gradients

An effective attribution-based method for explaining large language models.

![image](https://github.com/swarnava-sr/UDIG/assets/82533666/0cd31f2f-da8b-49a2-888c-5fa516805a37)

Interpolation paths used by UDIG. W is the word of interest and W` is the baseline. The green straight line represents the linear path used by IG for calculating attribution.

- **(left) UDIG-Greedy**: Grey regions are the neighborhood of the points chosen on the straight line. Each word in this neighborhood is first monotonized where each red arrow signifies the distance between the word and its corresponding monotonic point. The word closest to its corresponding monotonic form is selected as the anchor word (w4 since the red arrow of w4 has the smallest magnitude).

- **(right) UDIG-Max-Count**: The word with the highest number of monotonic dimensions (count shown in []) is selected as the anchor word (w3 since it has the highest number of monotonic dimensions), which is followed by updating the non-monotonic dimensions to make it monotonic - c1 (red arrow). Repeating this process multiple times for each point gives the non-linear blue path for UDIG.

## Installation

### From Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/king17pvp/UDIG.git
cd UDIG
pip install -e .
```

### Dependencies

The package requires:
- Python >= 3.7
- PyTorch
- Transformers >= 4.26.1
- Captum >= 0.8.0
- NumPy
- Scikit-learn
- tqdm

## Usage

```python
import udig
from udig import DiscretetizedIntegratedGradients, calculate_attributions

# Create UDIG attribution function
attr_func = DiscretetizedIntegratedGradients(forward_func)

# Calculate attributions
log_odds, comprehensiveness, sufficiency, attributions, delta = calculate_attributions(
    model, tokenizer, inputs, device, attr_func, base_token_emb, nn_forward_func, get_tokens
)
```

## Package Structure

- `udig.dig`: Core DiscretetizedIntegratedGradients implementation
- `udig.udig`: Main UDIG functions and utilities
- `udig.attributions`: Attribution computation functions
- `udig.monotonic_paths`: Monotonic path generation algorithms
- `udig.xai_metrics`: XAI evaluation metrics

## Citation

If you use UDIG in your research, please cite the original paper:

```bibtex
@misc{roy2024uniformdiscretizedintegratedgradients,
    title={Uniform Discretized Integrated Gradients: An effective attribution based method for explaining large language models}, 
    author={Swarnava Sinha Roy and Ayan Kundu},
    year={2024},
    eprint={2412.03886},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2412.03886}, 
}
```


