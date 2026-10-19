# Claude Code Instructions

Read [LIGHTCURVELYNX_GUIDE.md](LIGHTCURVELYNX_GUIDE.md) before making any changes. It is the canonical reference for design principles, core abstractions, API conventions, and development workflow.

SO for LYNX
=============
The aim of this project is 
General
-------
- For any new modules, you can initialize a git repository and commit as needed to track changes
- Install and use uv to manage Python packages
- uv pip install pixell and use functions from it
- Write reports to html that include figures from test plots. Always update the relevant report as the project progresses.
Coding guidelines
-----------------
- Reduce code duplication by reusing repeated parts
- Write tight, concise, readable code
- Avoid type hinting and other distractions
- Write full docstrings for each function
- Any scripts you write (for testing or otherwise) should have argparse CLI and a --debug option for quick tests (e.g. with less number of simulations or objects and/or lower resolution)
- Run ruff linting and formatting always before running code
Projcet Specific
----------------
- You are in a directory for a code base which simulates lightcurves for various astrophysical transients and them simulates observing them with upcoming or existing surveys. I would like to add a new survey. The surveys are controlled by the SurveyInfo objects. They require passbands, a noise model, and an observation table. The new survey is a Cosmic Microwave Background observatory. I can easily provide the passbands and noise model. The observation table however is an issue. The observation table assumes the survey is an optical instrument; the telescope will point at some point in the sky, take an image, and then move on to another patch. As such it assumes that the table is populated with individual pointings. The CMB observatory I would like to add instead scans the sky continuously.

- SO Observes the entire sky that is visible from 22°57′31″S 67°47′15″W.. The telescope scans constantly in azimuth at a fixed elevation of 40 degrees. The azimuth throws are 120 degrees. Half of the scans are between 30-150 az and the other half are between 210-330. The scan speed is 1 degree per second. There are actually 13 maps produced daily, one for each optics tube in SO. The field of view for an individual OT is ~1.25 degrees diameter, and they are center separated by ~1.75 degrees. They are in a hex patern, with half of the outer tubes missing. Eight are mid frequency, four are ultra high frequency, and one is low frequency.

- Some code has already been written to do this. It lives in obstable.simons_observatory_obstable. It is unclear if this code works but it is a good starting point.