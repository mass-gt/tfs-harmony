Methodology
===========

The Tactical Freight Simulator (TFS) implements an agent-based, rule-based
freight model. This page summarises the core modelling choices.

Choice models
-------------

**Multinomial Logit (MNL)** models are used for:

- Vehicle-type choice in distribution and tour formation.
- Parcels-mode choice (crowd-shipping vs. van vs. micro-hub).
- Logistic-segment choice.

Utilities follow the standard form:

.. math::

   U_{ni} = V_{ni} + \varepsilon_{ni}

where :math:V_{ni} = \beta^\top x_{ni} is the systematic utility and
:math:\varepsilon_{ni} is an i.i.d. Gumbel error term, giving the
MNL choice probability:

.. math::

   P_{ni} = \frac{e^{V_{ni}}}{\sum_j e^{V_{nj}}}

Gravity models
--------------

Freight generation uses a production-constrained or
attraction-constrained gravity model:

.. math::

   T_{ij} = A_i O_i \frac{e^{-\beta c_{ij}}}{\sum_k e^{-\beta c_{ik}}} D_j

where :math:T_{ij} is tonnes shipped from zone *i* to zone *j*,
:math:O_i and :math:D_j are productions and attractions, and
:math:c_{ij} is the generalised cost (distance or time).

Tour formation
--------------

Shipments are assigned to vehicle tours using two heuristics:

1. **Savings algorithm** — merges nearby shipments to minimise total distance,
   exploiting the classic Clarke-Wright savings metric:

   .. math::

      s_{ij} = d_{0i} + d_{0j} - d_{ij} - d_{0i,j}^{merged}

2. **2-opt improvement** — local search that removes and reinserts tour segments
   to eliminate crossing routes.

Random draws
------------

Stochastic elements (logit errors, parcel success rates, crowd-shipping
adoption) use a fixed `numpy` random seed (`SEEDS` parameter) to guarantee
reproducibility. Seeds are set at the start of each module.

For full mathematical detail, see the project documentation PDF:
`docs/Documentation MASS GT v3.pdf`.
