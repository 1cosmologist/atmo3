# AGENTS.md 

## atmo3

### Package description
The atmosphere is partially transparent to electro-magnetic radiation between radio frequencies of about 1 GHz
to mm-wave/far-IR frequencies of about 1000 GHz. Ground-based telescopes that operate in this range of frequencies
are impacted by the emission and absorption due to the intervening layers of the atmosphere. The primary component 
of atmosphere that impacts the atmospheric emission is the water vapor component and its fluctuations. Next biggest
component is the oxygen emission due to temperature fluctuations. The atmo3 - atmo(sphere) cube package is a box 
simulation tool to produce synthetic fluctuations of temperature and water vapor in a box of size of few tens of 
kilometres. The code then simulates observation by a telescope through the atmosphere along some line-of-sight. 
The emission module converts the temperature and water vapor along scans to brightness temperature following the 
physics of water vapor and oxygen emission. 

### Coding guidance
Do not write unnecessary lines of code. Do not add additional boilerplate code. Always present an ouline of the plan 
for any code change or new code. Get user feedback on the plan before executing. Always track physical units of
variables throughout the project. Make sure the physical principles and assumptions made in the project are consistent
and do not change assumptions without explicit user guidance. Always check to keep memory footprint minimum while not 
sacrificing speed. 

