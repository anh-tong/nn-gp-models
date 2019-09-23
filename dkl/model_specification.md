# Description

The implementation is mainly based on the following paper

```
@inproceedings{wilson2016stochastic,
  title={Stochastic variational deep kernel learning},
  author={Wilson, Andrew G and Hu, Zhiting and Salakhutdinov, Ruslan R and Xing, Eric P},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2586--2594},
  year={2016}
}
```

Let $g(x; w)$ be a neural network where $w$ are weights. The model tries to learn GP latent function $f$:

$$f(x) \sim \mathcal{GP}(0, k(g(x; w), g(x'; w)))$$

Typical problems for this model are classification problems where we don't have tractable inference for this model. 
Also, to scale GP for large-sized data, sparse GP with variational inference is a common approach.

Roughly speaking, sparse GP introduces a set of pseudo (inducing) data points which are assumed to have joint Gaussian distribution 
with data. We have to learn GP hyperparameters as well as these inducings. The vartional method approximates the posterior of the joint of
 the GP latent function $f$ and the inducing points $u$. By defining the variational the the form
  
  $$q(f,u) = q(u)p(f|u)$$
 
where $q(u) = \mathcal{N}(\mu, \Sigma)$ and $p(f|u)$ is just the Gaussian conditional distribution. We now can write the 
evidence lower bound in a nice form. The learning is done by stochastic gradient descent. Further detail is in 
 
```
@inproceedings{hensman2013gaussian,
  title = {{G}aussian processes for big data},
  author = {Hensman, James and Fusi, Nicolo and Lawrence, Neil D},
  booktitle = {Conference on Uncertainty in Artificial Intellegence},
  pages = {282--290},
  year = {2013},
  organization = {auai.org}
}
```
