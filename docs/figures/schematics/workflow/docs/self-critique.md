# Faithfulness pass (step 9, standard depth) and fixes

| Check | Drawing | Source | Result |
|---|---|---|---|
| Rim held at rim Fo | every model curve starts at segment a's Fo at distance 0 | note: edge 1 held at edge Fo | OK |
| Segment order | a at the rim, b inward | note: a, b, ... from edge 1 inward | OK |
| Distance origin | "distance from rim", rim at left in every profile plot | note: distances from the profile's own edge | OK |
| Spot count | crystal traverse had 9 spots, plot 13 points | P28 | **fixed**: traverse draws the same 13 |
| Time direction | arrow in b meets t1, t2, t3 in order; core-side Fo falls with time | erfc solution | OK |
| Misfit | high at t1, minimum at t2 (best fit), rising at t3; 1σ band around the minimum | note fit steps 2-3 | OK (Δχ² = 1 used; FORSTER uses a slightly larger small-sample threshold: caption) |
| Noise model | residuals at best fit = white scatter + smooth correlated part | note | OK |
| Starting-profile priors | on step position and segment Fo, fitted with the time; marked optional | note step 4 | OK; the bold step in d is the nominal one (caption) |
| Monte Carlo | draws T, P, fO2 and the law's parameters (not a switch between the four laws) | README mc_law | OK |
| Inverse mapping | each D draw at +δ maps to t at −δ (log) | t ∝ 1/D | OK |
| Nesting | fit alone ⊂ + starting profile ⊂ + conditions and law; median and 68% from the last | plotting.py | OK (computed, not hand drawn) |
| Orientation | generic triad, not tied to crystal faces | note: habit unknown | OK; added the word "orientation" so the triad reads as an input |
| Must-not list | no numbers, no habit, no growth stages, no joint fit, no code names | brief | OK |

Rubric (1-5): reads at size 4 (6 pt labels, gate warns they are small for 183 mm); takeaway 4 (D·t is shown
where it acts, √(D t) in b and t = (D t)_fit / D in e, but there is no single headline mark for it); faithful 5;
correct 5; links 4 (t1-t3 colours b→c, blue best fit b→c→d, three source colours d/e→f); economy 4 (98 words,
0.48 words/cm², slightly over calibration); craft 4.
