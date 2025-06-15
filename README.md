# A collection of data-driven models for forecasting


URL: https://github.com/evgenykurbatov/ddforecast

Here, a collection of models for forecasting of numerical (i.e. not categorical) sequences.


## Models

**SimpleTransformer** is essentially a stack of transformer-encoder layers, literally the `torch.nn.TransformerEncoder`, provided with linear layers for mapping the input and output to and from an embedding space. Simple sinusoidal positional encoding is used here.

**RoPETransformer** generally resembles the `SimpleTransformer` architecture, but uses RoPE positional encoding.


## Author

**Evgeny P. Kurbatov**

- <evgeny.p.kurbatov@gmail.com>
