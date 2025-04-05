# FAQ

## General

### What is a catalyst? What is a bulk? What is a surface? What is an adsorbate? 

* **Catalyst:** A catalyst is a material that makes a chemical reaction happen faster without itself being consumed. The catalysts shown here are one specific type of catalyst (heterogeneous) catalyst where the reactions happen as molecules interact with a catalyst surface.   
* **Bulk:** An inorganic crystal structure that will form the base structure of our catalyst surface and from which we will identify possible surfaces.  
* **Surface:** A surface formed from a bulk crystal structure by cutting along a specific plane (specified by the Miller index)  
* **Adsorbate:** The small organic molecule interacting with the catalyst surface that is adsorbed on the catalyst surface.

### What are these models and simulations useful for? 

Simulations like these are used to design materials (catalysts) that make chemical transportations more efficient\! Example uses of catalysis in day-to-day life:

* The catalytic converter in your car is a special catalyst that breaks down harmful automotive exhaust to more benign species  
* Catalysts are responsible for taking nitrogen in the air and making ammonia for fertilizer and agricultural use. We couldn’t feed the world's population with catalysts\!  
* Energy can be stored as renewable hydrogen produced by splitting water over electrocatalysts  
* Fuel cell vehicles rely on catalysts to combine hydrogen and oxygen into water while producing electricity  
* Carbon capture systems can use catalysts to turn waste or captured CO2 into useful chemicals

These simulations can help us design cheaper, more efficient, and more durable catalysts. These simulations can also help design catalysts for new chemical reactions that are currently difficult or infeasible to drive at scale. These simulations could also be used to design materials that are resistant to surface oxidation or corrosion. 

### Why does Meta care about catalysis?

* Catalysis is important for climate/renewable energy and achieving net-zero carbon emissions in the near future. Meta has a number of decarbonization goals that rely on societal solutions for renewable energy and energy storage. Catalysis is key to many of these technologies, and helping the community develop more efficient processes will help us reach those goals\!  
* Meta Fundamental AI Research (FAIR) is interested in pushing forward the state-of-the-art in many applications of AI/ML to the world, including vision, language, robotics, etc. Applications of AI in science and chemistry is one such exciting area of research.  
* The core models powering this API are from a class of models known as [graph neural networks](https://distill.pub/2021/gnn-intro/) (GNNs). GNNs are an exciting area of research in AI.

### Are there any rate limits on incoming requests?

TODO

### How do I know which surface/adsorbate/etc to pick when running this?

**If you are new to catalysis:**

* Try selecting some simple (one element) or complicated (multiple elements) bulk structures.   
* Try selecting some small and larger adsorbates to see how interesting the structures may be  
* Watch some of the relaxations and see how subtle some of these relaxations can be\!  
* Develop some intuition for how the energies depend on different surfaces and bulk structures

If you have ideas on how to build better ML models to predict the final energies or relaxation pathways, you should visit opencatalystproject.org\! Hopefully this website gives you an idea of the challenge here and what the data will look like. 

**If you are an experienced catalyst researcher:**

* You might want to see how common descriptors for your favorite reactions (like \*CO or \*CH2) might vary across different surfaces or compositions  
* If you are interested in reaction pathways for a single surface, you might want to predict properties for several adsorbates on the same surface  
* Let us know if you have other suggestions here\! We’d love to hear what you’re using this for\!

Hopefully this gives you an idea of what the state of the art is for reactive catalyst potentials\! The world is your oyster\!

### What are the black lines and boxes in all of the visualizations? 

The systems considered here are periodic, i.e. they repeat over and over infinitely in the X/Y direction to approximate a very large catalyst surface. The black lines show the edges of the repeating unit cell, or [periodic boundary conditions](https://en.wikipedia.org/wiki/Periodic_boundary_conditions). 

### Why do only some of the atoms move?

These structures are an approximation of small molecule intermediates interacting with a much larger nanoparticle surface. Even small nanoparticle catalysts will have many layers to the surface. It is very common to freeze the bottom few layers of the catalyst surface for these simulations so that they do not move and are more similar to the actual much deeper structure. You will notice mostly the small molecule moving during the relaxation, and you may also notice small movements of the top layer of surface atoms as they compensate for the adsorbate. 

### Why is the adsorbate broken up / appearing on two sides of the cell?

We are working to fix this\!

## ML questions

### What sort of models are used to generate these predictions?

The state-of-the-art in ML for chemistry is moving extremely quickly, and we track this exciting progress through the open leaderboards available at https://opencatalystproject.org/leaderboard.html. The models that are currently available are openly available models that are near the top of the leaderboard and have a desirable compute/accuracy trade-off. We are excited to see developments in many ML model types, but the ones that have worked best for large catalyst datasets tend to be message passing or graph neural networks. 

### How big are these models? 

The models used here have \>100 million free fitted parameters that are fit to the OC20 and OC22 datasets. These datasets have \>100 million structures, each with O(30) energy/forces, so the number of targets available to fit is quite massive (in the context of work in AI for science).

### Can I run these models on my own computer? How long would it take?

Yes\! All of the models and pre-trained checkpoints are open source and freely available at [https://github.com/Open-Catalyst-Project/ocp](https://github.com/Open-Catalyst-Project/ocp). They can run on CPUs with \~16Gb of RAM, and CUDA-compatible GPUs with \>16Gb of memory. Each energy/force call usually takes O(1s) on a cpu core, and O(50ms) on a GPU, averaged over a reasonable batch size. Of course, this depends on your precise setup and your mileage may vary.

### What about the CO2 emissions associated with training and serving ML models? I heard those were significant. 

Very relevant question\! Greenhouse gas emissions for training and using large ML models like those used on this website are an active research area. This includes how to measure them, and how to reduce them using methods like inference accelerators or more efficient training strategies. In this case, the models that we are replacing are even more resource intensive – the density function theory (DFT) calculations that would typically be required for simulations like those shown often require 1000s of core-hours to analyze many possible configurations. The emissions associated with using the ML models here are a small fraction of what would be emitted while running traditional DFT simulations. We also hope that we will see community progress in developing more energy-efficient models with similar (or better) accuracies to the models shown here\!

### I ran a prediction but the structures/energies look strange to me. What should I do?

Please let us know by posting as a [github issue](https://github.com/FAIR-Chem/fairchem/issues)  with your inputs, results page link, and structure so we can look into the problem on our end.

Alternatively, a few things to try:

* First, try a couple of the ML models available on this website for the same surface/adsorbate and see if the results differ. This gives you an idea of whether it is specific to a model, or something about the surface/adsorbate that leads to problems.  
* Second, you can download the structures and try other ML models from [https://github.com/Open-Catalyst-Project/ocp](https://github.com/Open-Catalyst-Project/ocp) to see if the problems are consistent.   
* Finally, if you have access to VASP you can try running the relaxations yourself to verify the results.

### Do you have any estimates on how much I should trust these predictions? Why aren’t there error bars? 

Uncertainty quantification for large GNN models like those used here is a very active research area, with many different uncertainty calibration metrics and varying additional compute costs for the prediction. 

* You can get some idea of average MAE from the OCP leaderboard. The models used here have MAEs across many different catalyst/adsorbate systems of \~0.3 eV. For metals and smaller adsorbates, the residuals tend to be smaller.  \[SEE LINK\]  
* Deciding which configuration corresponds to the minimum energy tends to be more robust to residuals, so it is likely that the identified structure is identified. Recent results on validation datasets suggest that approximately 50% of the time the energy identified in this process is within 0.05 eV of the DFT-computed \[SEE ADSORBML\] 

We are investigating efficient ways to add uncertainty to these predictions; more updates coming soon\!

## Catalysis questions

### All I know is the rough composition of the material I’m interested in. How do I select a bulk structure or surface in the website?

Generally, materials with lower formation energies (or smaller distances to the lower hull of the phase diagram) are more likely to be stable or metastable. The Materials Project and other databases have a bunch of great tools to help predict bulk stability (including under reaction conditions\!). 

For surface structure, you have a few options:

* You could see how your properties of interest vary across similar compositions or surfaces to get a feel for how surface structure or composition might impact reactivity  
* You could use DFT or other ML models to predict surface stability to help you select a specific surface  
* You could find a computational chemistry catalyst friend/collaborator to dig into the surface structure\!

### How do I use these predictions to understand or predict the activity/selectivity of a catalyst?

The catalyst predictions exposed on this service can be used in many ways; we’ve tried to highlight a few potential use cases in the \[EXAMPLES SECTION\]. A few possible use cases:

* If you or others have already identified ideal adsorption energies for a particular chemistry, you can use this service to compare the adsorption energies across several facets on the catalyst, or compare different catalyst surfaces.   
* The adsorption energy on different sites of a particular surface may help you identify which surface is responsible for catalytic activity.   
* If you know the catalyst surface you are interested in, you could use these predictions to construct reaction energies and a free energy diagram to compare different reaction pathways. You could also do this using your favorite microkinetic modeling package\!  
* You could use these calculations as a starting point for vibration calculations (e.g. IR spectra) or reaction activation barriers (transition states),

If you find these useful for your work, we’d love to highlight some additional use cases/stories, so please reach out via email\! \[LINK\[

### Stability/durability is important for my application; can I predict that?

Yes, this is something you can predict, but probably using different ML models than the ones trained here. A few great options:

* The formation energy, energy above the phase diagram hull in Materials Project and similar databases are popular descriptors for whether a material is stable, metastable, or unstable. Reaction conditions can be included via techniques like Pourbaix diagrams.  
* There are many machine learning models available to predict stability for arbitrary materials.

The stability of more complex catalyst interfaces is a very interesting research area, especially as some materials are self-passivating and stable even under conditions where they decompose. 

### I made a catalyst with a different crystal structure than is present in the drop-down list; what should I do?

If you know the crystal structure of composition of your catalyst but it’s not present in the drop-down list, it’s probably because the structure is either not in the Materials Project, or it’s predicted to be unstable by more than 0.1 eV/atom, or our calculations failed when we relaxed the inputs with DFT/RPBE. 

* You can use the new [Open Catalyst API](ocpapi) to enumerate surfaces and perform the adsorbate placement using python or your web browser. Make sure you have an RPBE-relaxed structure before starting this process\!  
* You can also do these by hand using the [Open Catalyst Project tools](tutorials/NRR/NRR_example)

### I think my material is more complex than the surfaces shown here (surface segregation, additional terminations, etc); what should I do?

The models used here may be able to predict the adsorption on more complex surfaces (for example, solid solutions, single atom alloys, segregated materials, etc), but the models have not been validated in these situations. We recommend downloading and using the pre-trained models on your own machine using the ASE calculator interface and using them to predict the adsorption energy [calculations/adsorption_energies]. If you find the models work well for your application, we’d love to hear from you\! And if they don’t, feel free to reach out via a [github issue](https://github.com/FAIR-Chem/fairchem/issues).

We’re considering allowing predictions on more diverse surfaces, but there are some computational nuances that make doing so a bit difficult. Feel free to email the team \[EMAIL ADDRESS\] if you have a specific use case or are interested in getting updates\!

### I’m interested in electrochemistry, but I don’t see any solvent effects or water layer; what should I do?

It’s very common in the electrocatalyst modeling community to relate gas-phase adsorption energies (like those shown here) to adsorption energies in solvent by incorporating a per-adsorbate solvent correction to the adsorption energy. This correction tends to be largest for adsorbates that can hydrogen bond with a water layer (like \*OH), or adsorbate that have a strong dipole moment that can be screened with a water layer. This approximation is helpful in screening millions of possible catalyst surfaces, especially because the structure of the water layers on each surface is difficult to predict.   
Of course, by not including a solvent the models cannot help predict the impact of the solvent on the catalyst activity/selectivity, the role of cation/anion additives in the solvent, or interesting enhancements/disturbances to proton transport, or reactions from the solvent layer to surface adsorbates. These are all exciting research areas\! 

*Note: even using DFT, full fidelity detailed modeling of the solvent/catalyst interface in electrochemical conditions is an active research challenge!

### I’m interested in trying this across many different surfaces or discover new catalysts but it’s hard to select them all on the website; how should I try this? 

This website is meant to acquaint you with the state-of-the-art in accelerated catalyst property prediction models, and demonstrate the capabilities of models trained on the Open Catalyst Project datasets. If you want to use these models for high-throughput simulations or programmatically across several materials, you are welcome to:

* Use Open Catalyst Project API (python or REST); note that this may be rate-limited  
* Download and use the Open Catalyst Project models and toolkits (all open source and permissively licensed\!) to predict properties on many systems

If either of these are unclear or you have trouble using them, please direct questions to the github repo!

## Computational catalysis questions

### What are the caveats of using these calculations?

As the statistician George Box once said, “*All models are wrong, but some are useful\!*” 

Simulations like these are extremely common in the catalysis community to develop intuition into limitations of specific catalysts, propose new catalyst modifications or active sites, or discover new catalysts with interesting reactivity/selectivity. However, there are a number of caveats to be aware of:

* These simulations are based on Density Functional Theory, a quantum mechanical simulation technique that offers reasonably predictive properties, but is nonetheless an approximation.   
* The DFT settings in the training dataset were chosen for a reasonable combination of speed/accuracy trade-off (kpoints, energy cutoffs, pseudopotential, etc).  
* All simulations neglected spin polarization (magnetic moments), which often have a small impact on adsorption energies but may be important. Of course, this only applies to spin-polarized elements like Ni, Fe, etc.   
* The simulations neglect long-range dispersion interactions, which are more important for large adsorbates than small ones.   
* These simulations approximate a complex catalyst surface with a very small and defect-free representation of an active site. Real catalysts may undergo restructuring or segregation under reaction conditions. Solid solution materials are also not included in databases like the Materials Project so are not shown here.  
* These simulations do not include solvent interactions, which may be important for your application. However, adsorption energies with and without solvent are often strongly correlated, so this doesn’t mean the calculations can’t be used for catalysis in condensed phases\!  
* The reaction energies here are enthalpies, and do not include entropic contributions to stability (as they would be if they were the Gibbs free energy).   
* The ML models shown are imperfect and rapidly developing. The mean absolute error for the OC20 validation datasets is \~0.3-4 eV (less for metals and small molecules, more for complex catalysts like nitrides or larger adsorbates) \[CITE leaderboard\]. The best adsorption site according to DFT is one of the best 5 predicted by these methods about 90% of the time.

These caveats are all extremely common in computational catalysis, and are provided here not to scare you but to remind you of possible limitations. If you aren’t sure whether these caveats are important for your intended use case, talk to your favorite computational catalyst researcher friend, or post your questions on the discussion board\!

### I’m interested in a different reaction energy than the ones listed for a particular adsorbate; what do I do?

The reaction energies listed on this website are helpful because they all use a single set of reference gas phase species. This makes it easy to mix and match reaction energies to make new reaction energies, perhaps using gas phase species. 

For example, adsorption of \*NH3 on a surface would show up on this site as:  
\* \+ ½ N2(g) \+ 3/2 H2(g)-\> \*NH3  
If you wanted to know the adsorption energy of \*NH3 relative to gas phase NH3, we could add this reaction energy with the (known) gas phase formation energy of ammonia:  
\* \+ ½ N2(g) \+ 3/2 H2(g)-\> \*NH3           (E1)    
 NH3(g) \-\> 1/2N2(g) \+ 3/2H2(g)                    (E2)  
To yield (remember intro stoichiometry chemistry\!)  
\* \+NH3(g)-\> \*NH3                                             (E1+ E2)

Similar approaches can be used to prepare surface reaction energies. For example, if we wanted to know the reaction energy for  
\*CCH2 \+ \*-\> \*C \+\*CH2  
We could do that by adding the energies for the following three reactions (all available from the website\!):  
\*CCH2 \-\> 2CO(g)+3H2(g)-2H2O(g)+\*      (E1)  
\* \+ CO(g)+H2(g)-H2O(g)-\> \*C                  (E2)  
\* \+ CO(g)+2H2(g)-H2O(g)-\> \*CH2          (E3)   
Adding these three reactions together we get (notice all the gas phase energies cancel, neat\!):  
\*CCH2 \+ \*-\> \*C \+\*CH2                                    (E1+ E2+E3) 

### Why are the reaction energies enthalpies and not free energies? How should I calculate dG?

Calculating the free energy of reaction requires several additional steps in a typical computational catalysis workflow:

1. Vibrational calculations to obtain vibrational modes and energies  
2. Calculation of the zero point energy (ZPE) and entropic contributions (assuming harmonic or anharmonic vibrational modes)  
3. (optional) identification of adsorbate-specific solvent or frequency-dependent corrections  
4. Calculating the Gibbs free energy at a specific temperature

We expect that the ML models used here can also help predict vibrational modes and are working to validate these approaches. As a placeholder, we have provided rough estimates for Gibbs free energy of reaction for the reactions here by calculating corrections for some representative simple metal surfaces. These free energy corrections are usually fairly constant across different catalyst surfaces, but can change significantly if the binding configuration (mono- or bi-dentate) or active site changes. 

### Where did the list of bulk materials come from? How did you select them from the Materials Project? Why can’t I use my own materials?

The bulk materials for this demo were prepared by:

1. Selecting all materials from the Materials Project using catalyst elements seen in OC20  
2. Filtering the materials to negative formation energies and energies above the phase diagram hull less than 0.1 eV/atom  
3. Isotropic relaxations of the structures with DFT and the RPBE functional, which had a small failure rate.

We’re interested in adding the ability to upload custom bulk structures, but making sure the materials are not strained from the perspective of the RPBE functional is a bit nuanced and it’s hard to know how well our models will work on very different structures. If you’re interested in this feature, reach out via a [github issue](https://github.com/FAIR-Chem/fairchem/issues)!

### Where did the list of adsorbates come from? How did you select them? Why can’t you select or add your own?

These adsorbates were the original 82 adsorbates used to construct OC20, which allows us to have some intuition and statistics on how well the models may work. We’re considering allowing users to upload custom adsorbate structures. If you’re interested in this feature, reach out as a [github issue](https://github.com/FAIR-Chem/fairchem/issues)!  

### My favorite catalyst composition/surface/adsorbate is missing. What should I do?

This website is meant to acquaint you with the state-of-the-art in accelerated catalyst property prediction models, and demonstrate the capabilities of models trained on the Open Catalyst Project datasets. If you want to use these models for high-throughput simulations or programmatically across several materials, you are welcome to:

* Use Open Catalyst Project API (python or REST); note that this may be rate-limited  
* Download and use the Open Catalyst Project models and toolkits (all open source and permissively licensed\!) to predict properties on many systems

If either of these are unclear or you have trouble using them, please direct questions to the discussion board\! \[LINK\]

### These are all low coverage energies, but I’m interested in adsorption energies at high coverage. Can we do that too?

This is a very exciting and interesting research area\! We expect that OCP models like those shown here will be capable of predicting energies at higher coverages, but this is an ongoing research area.  This is especially interesting for adsorbates with long-range interactions like \*CO, which has a strong dipole moment. More training data or more sophisticated models may be needed to accurately resolve high-coverage adsorption energies.

### Reaction energies are great, but I’m interested in reaction kinetics (activation energies). Can I use OCP models for kinetics?

This is a very exciting and interesting research area\! We expect that OCP models like those shown here will be capable of predicting activation energies and reaction barriers, but this is an ongoing research area. More training data or more sophisticated models may be needed to accurately resolve transition state energies. 

### What DFT settings should I use to verify the single-points? How would I reproduce these energies with DFT?

The [Open Catalyst Dataset repo](https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset/tree/main) can be used to create DFT inputs. Specifically, the DFT settings are provided [here](https://github.com/Open-Catalyst-Project/Open-Catalyst-Dataset/blob/main/ocdata/utils/vasp.py#L20-L35) to ensure consistency with the underlying OC20 DFT-level theory.


### What does the “shift” mean in the surface information? 

The shift here is the absolute position of the cut and the resulting termination, and is used internally in pymatgen \[LINK to pymatgen\] for producing the  surfaces. For very simple surfaces (e.g. Pt(111) or Cu(211)) there is only one unique surface termination so we don’t need to specify the shift. However, more complicated materials such as bimetallic systems may have several unique terminations that we can specify with the shift. It is sometimes possible to identify the best possible termination, but for bimetallic surfaces this is often not obvious so we include all of the possibilities here. 

### I found a better adsorbate configuration than the one shown on this website; what should I do? 

This is expected to happen when one of the following happens:

* Our adsorbate placement / guessing strategy isn’t exhaustive enough to find the best possible configuration  
* The trained ML models encounter errors or produce bad relaxations

Either way, post on the discussion board so we can improve the models and datasets\!