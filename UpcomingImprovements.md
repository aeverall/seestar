## GitHub account:

- Include data for RAVE-TGAS so that selection function can be run.
- Create an example running code script.
- Include Plate selectors for 2MASS/Vista data.
- Fix scripts so that the github account is running.

## Local:

- Create local ipynb which tests the operation of the code.

## Code improvements:

- Generalise so that both 2MASS and Vista are included in the survey.
- Create point distributions so that Poisson noise factors are minimised.
- Create optional Union calculation for overlapping fields.
	- If final survey has star double counts removed, take sum
	- If final survey has star double counts still in, take Union (area normalised)
- Parallel computing - decide how to break down code based on number of available cores.
