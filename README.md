Baseline basic transformer model vaguely modeled after llama 3. This repo is primarily for refrence but the model is trainable:

To train on all of tinystories-hf for one epoch:
```python
uv run python main.py
```

Inference on the resulting model:
```
uv run python basic_inf.py
```

The rest of this is going to detail the general premise of the architectural points and my thoughts on them.

![Architecture](architecture.svg)
