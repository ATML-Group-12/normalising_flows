# Variational Inference with Normalizing Flows


__TLDR__: We perform variational inference by chaining many layers of invertible transformations together. The invertible transformations mean that we can compute functions of the expressive densities which we do not have explicit access to using the law of the unconscious statistician.

- Computation of the Jacobian of these transformations can have cubic complexity and in the paper they find a family of invertible transformations which have linear complexity for computing it.
- As the depth of the flow tends to infinity we end with an ODE
- In the paper they mention the "Free Energy" which is a term for the Evidence Based Lower Bound _ELBO_, which is a proxy objective which by optimizing it enables us to minimise the KL Divergence between the true posterior and the variational distribution.



Algorithm
- Construct a Network
- The first layer generates a distribution conditional on the data (this distribution should be well understood i.e probably an MVN)
- The next layers are simply invertable transformations.
- Our object we are optimising is the ELBO, and with each minibatch we optimise for this.


Extension Ideas:
- Natural Gradient Descent