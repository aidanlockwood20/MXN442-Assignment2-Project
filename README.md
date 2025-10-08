# MXN442 - Assignment 2 Project Code

---

## Preparing the working environment

All packages that I (Aidan) have used in the project is outlined within the `requirements.txt` file. Ensure to setup a python environment using `python -m venv <name of the environment you want to create>` and then run:

pip/pip3 (depending on how you do pip installs on your local machine) install -r requirements.txt

Once this step has been completed, you should be able to use the notebook and add changes accordingly.

### Obtaining the UNSPSC Product Codes

---

The UNSPSC Product Codes which assist in filtering the observations down to pharmaceutical-only observations is in this [link here.](https://www.ungm.org/public/unspsc)


### Things to consider for the report
- Whether or not we can access the other country tender data. If not, the UN codes are redundant 
- Once we calculate the unit prices, we need to apply an inflation factor to each item for that year
- Need to apply currency conversion to the unit/total prices for the tenders

- Run a sample of the dataset for us to run our models on 


### Currency Reporting
1. Calculate the per-unit prices for each item in the dataset - using the quantities found from the tender API
2. Decide on a figure to adjust all unit prices to (with respect to inflation) - I would say 30 June 2025 for consistency
3. Calculate those unit prices for each year and place that into a separate column 
4. Use this calculated column for modelling 
5. Retrieve output from the model
6. Report both the national currency AND either/and USD and AUD

