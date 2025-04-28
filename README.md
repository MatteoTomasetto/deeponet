# Deep Operator Networks

This directory contains an implementation of *Deep Operator Networks* for [CTF-for-Science](https://github.com/CTF-for-Science).
*Deep Operator Networks (DeepONets)* are a class of neural operator architectures designed to learn mappings between infinite-dimensional function spaces. For a complete presentation see, for instance, [1,2]. Specifically, DeepONets decompose an operator into two cooperating sub-networks:
- *Branch net* that encodes input functions at a finite set of sensors,
- *Trunk net* that encodes the coordinates at which the output function is evaluated.
In formula, the operator $G: V \to U$ between infinite-dimensional function spaces $V$ and $U$ is approximated though the product

$$ G(v)(\eta) = \boldsymbol{b}(v) \cdot \boldsymbol{t}(\eta) $$

where $\mathbf{b}(v)$ is the branch net output dependent on the input $v \in V$ (finite dimensional input are typically considered relying on a finite set of $n$ sensor measurements $\mathbf{v} \in \mathbb{R}^n$ of the function $v$), and $\mathbf{t}(\eta)$ is the trunk net output dependent on the coordinates $\eta$.

For instance, when dealing with time-series data as taken into account by [CTF-for-Science](https://github.com/CTF-for-Science), it is possible to consider the operator

$$ G(u_{t-1},...,u_{t-k})(\eta) = u_t(\eta) \approx \mathbf{b}(\mathbf{u}_{t-1},...,\mathbf{u}_{t-k}) \cdot \mathbf(t)(\eta) $$

where $k$ is the lag parameter and $\eta$ is the spatial coordinate where to predict the evolution of the function $u$. As proposed by [3], the time instance $t$ or the time-step $\Delta t$ may be added to the trunk input. 

## Files
- `deeponet.py`: Contains the `DeepONet` class implementing the model logic based on [DeepXDE](https://github.com/lululxvi/deepxde).
- `run.py`: Batch runner script for running the model across multiple sub-datasets.
- `config_*.yaml`: Configuration file for running the model.

## Usage

Run the model with:

```bash
python models/deeponet/run.py models/deeponet/config.yaml
```

## Dependencies
- Add any model-specific dependencies here.

## Description
- Add a detailed description of your DeepONet model, its parameters, and usage instructions here.
