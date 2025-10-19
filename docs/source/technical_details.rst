Technical Details
=================

This page provides technical details about FCVOpt's implementation and methodology, based on the paper `"Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms" <https://doi.org/10.1080/00401706.2025.2515926>`_.

Problem Motivation
----------------------

K-fold cross-validation (CV) is more robust than single holdout validation but computationally expensive since models must be trained K times at each hyperparameter configuration. The key insight is that fold holdout losses are pairwise correlated across different hyperparameter configurations due to overlap in training sets and common correlation structure.

True CV Loss Definition
---------------------------

The "true" CV loss function is defined as the expected value over all possible holdout sets:

.. math::

   f(\mathbf{x}) = \mathbb{E}[y_\mathcal{J}(\mathbf{x})]

where the expectation is over holdout set :math:`\mathcal{J}` chosen uniformly at random. This represents the CV loss averaged across all possible K-fold partitions.

Hierarchical Gaussian Process Model
---------------------------------------

FCVOpt uses a hierarchical Gaussian process (HGP) model where each observed fold holdout loss is decomposed as:

.. math::

   y_j(\mathbf{x}) = f(\mathbf{x}) + \delta_j(\mathbf{x}) + \epsilon_j(\mathbf{x})

where:

* :math:`f(\mathbf{x}) \sim \mathrm{GP}(\mu, c_f(\cdot, \cdot))` is the latent true CV loss function
* :math:`\delta_j(\mathbf{x}) \sim \mathrm{GP}(0, c_\delta(\cdot, \cdot))` captures fold-specific deviations
* :math:`\epsilon_j(\mathbf{x}) \sim \mathcal{N}(0, \sigma_\epsilon^2)` represents observation noise

Covariance Structure:

The covariance between fold holdout losses is:

.. math::

   \mathrm{Cov}(y_j(\mathbf{x}), y_{j'}(\mathbf{x}')) =
   \begin{cases}
   \sigma_f^2 \rho_f(\mathbf{x}-\mathbf{x}') + \beta\sigma_\delta^2 \rho_\delta(\mathbf{x}-\mathbf{x}') & j \neq j' \\
   \sigma_f^2 \rho_f(\mathbf{x}-\mathbf{x}') + \sigma_\delta^2 \rho_\delta(\mathbf{x}-\mathbf{x}') & j = j', \mathbf{x} \neq \mathbf{x}' \\
   \sigma_f^2 + \sigma_\delta^2 + \sigma_\epsilon^2 & j = j', \mathbf{x} = \mathbf{x}'
   \end{cases}

where :math:`\beta \in [0, 1)` is a cross-correlation parameter between different fold error GPs.

FCVOpt Algorithm
--------------------

The FCVOpt algorithm follows these sequential steps:

S0: Initial Design
  Generate initial design points :math:`\{(\mathbf{x}_i, j_i): i=1,\ldots,N_0\}` and observe losses.

S1: Model Fitting
  Fit the HGP model to data :math:`\mathcal{D}_N = \{(\mathbf{x}_i, j_i, y_i): i=1,\ldots,N\}` and find incumbent:

.. math::

   \mathbf{x}_{\text{inc}}^{(N)} = \arg\min_{\mathbf{x} \in \{\mathbf{x}_i: i=1,\ldots,N\}} \hat{f}_N(\mathbf{x})

S2: Acquisition
  Select next candidate configuration :math:`\mathbf{x}_{N+1}` by optimizing an acquisition function, then choose fold :math:`j_{N+1}` to minimize posterior variance.

Acquisition Functions:

Knowledge Gradient (KG):

.. math::

   \mathrm{KG}(\mathbf{x}|\mathcal{D}_N) = \min_{\mathbf{x}' \in \mathcal{X}} \mathbb{E}[f(\mathbf{x}')|\mathcal{D}_N] - \mathbb{E}[\min_{\mathbf{x}' \in \mathcal{X}} \mathbb{E}[f(\mathbf{x}')|\mathcal{D}_N, f(\mathbf{x})]]

Lower Confidence Bound (LCB):

.. math::

   \mathrm{LCB}_\kappa(\mathbf{x}|\mathcal{D}_N) = \hat{f}_N(\mathbf{x}) - \kappa \hat{\sigma}_N(\mathbf{x})

where :math:`\kappa` (e.g., :math:`\kappa = 2`) trades off exploitation vs exploration.

Fold Selection Strategy
---------------------------

After selecting candidate configuration :math:`\mathbf{x}_{N+1}`, the fold is chosen to minimize posterior variance:

.. math::

   j_{N+1}(\mathbf{x}_{N+1}) = \arg\min_{j \in \{1,\ldots,K\}} \mathrm{Var}[f(\mathbf{x}_{N+1}) | \mathcal{D}_N, \{(\mathbf{x}_{N+1}, j)\}]

This variance reduction criterion has important implications:

* Early stages: Selects previously-sampled folds to quickly rule out inferior regions
* Later stages: Selects unsampled folds near promising regions to improve accuracy
* Near sampled points: Avoids redundancy by selecting different folds than nearby configurations

Implementation Details
-------------------------

Kernel Functions:

FCVOpt uses the Matérn 5/2 kernel for numerical inputs:

.. math::

   c(\mathbf{x}, \mathbf{x}') = \sigma^2\left(1+\sqrt{5r^2(\mathbf{x},\mathbf{x}')} + \frac{5}{3}r^2(\mathbf{x},\mathbf{x}')\right)\exp\left(-\sqrt{5r^2(\mathbf{x},\mathbf{x}')}\right)

where :math:`r^2(\mathbf{x},\mathbf{x}') = \sum_{d=1}^{D} \frac{(x_d - x'_d)^2}{\ell_d^2}` and :math:`\ell_d` are length-scale parameters.


Parameter Estimation:

The HGP model has :math:`2D+5` hyperparameters:
* Mean :math:`\mu`
* Length-scales :math:`\boldsymbol{\ell}_f` and :math:`\boldsymbol{\ell}_\delta` (D each)
* Cross-correlation parameter :math:`\beta`
* Variance terms :math:`\sigma_f^2`, :math:`\sigma_\delta^2`, :math:`\sigma_\epsilon^2`

Parameters are estimated via maximum a posteriori (MAP) estimation.

Paper Citation
------------------

If you use FCVOpt in your research, please cite:

.. code-block:: bibtex

   @article{yerramilli2025fractional,
     title={Fractional cross-validation for optimizing hyperparameters of supervised learning algorithms},
     author={Yerramilli, Suraj and Apley, Daniel W},
     journal={Technometrics},
     year={2025},
     doi={10.1080/00401706.2025.2515926}
   }

