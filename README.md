https://drive.google.com/drive/folders/18tg9mTBZ3E6bmpnoelMbYqMo_o3B76bX?usp=sharing

# Efficient-Code-Generation-with-E-code


## How to Use

### Implementation Train the model -> predict the generated code -> perform IO test on the generated code.
#### To use the E_code source code extremely fast: 

1. Extract the GEC dataset to the E_code folder and change the file name to GEC. 
2. Run the train.py file. 

#### Fast-running classification experiments: 

Set Command_line_parameters.task = 0 to train the E-code model.

Set Command_line_parameters.task = 0 and set Command_line_parameters.RELU = 1 to train a comparison experiment using the RELU activation function.

Set Command_line_parameters.task = 0 and set Command_line_parameters. heads = 8 to train a comparison experiment using 8 heads.

Set Command_line_parameters.task = 1 to train the No-expert-E-code model.

Set Command_line_parameters.task = 2 to train the GPT model.

#### Extremely fast use of Time_Predictor source code: 
1. Extract the GEC dataset to the E_code folder and change the file name to GEC. 
2. Run the train.py file to train the model.

3. Put the code to be predicted into Code_to_be_predicted a
4. Run Prediction_generation_code to automatically predict the code runtime.




## Diagrammatic figure
In the Efficient-Code-Generation-with-E-code work, the diagrammatic figure is in the [Diagrammatic figure folder](https://github.com/CodeGeneration2/Diagrammatic-figure/tree/main/Diagrammatic%20figure).



### E-code 350M
We give [the results of 3 times code generation in the E-code 350M model](https://github.com/CodeGeneration2/Generated-code-has-been-predicted/tree/main/Generated-code-has-been-predicted/E-code%20350M).


### GPT-Neo 125M
We give [the case results of one code generation for the GPT-Neo 125M model](https://github.com/CodeGeneration2/Generated-code-has-been-predicted/tree/main/Generated-code-has-been-predicted/GPT-Neo%20125M).


### No expert group E-code 350M
We give [the case results of one code generation for the no expert group E-code 350M model](https://github.com/CodeGeneration2/Generated-code-has-been-predicted/tree/main/Generated-code-has-been-predicted/No%20expert%20group%20E-code%20350M).

