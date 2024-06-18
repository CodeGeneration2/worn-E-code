

# Efficient-Code-Generation-with-E-code

![E-code模型图](https://github.com/CodeGeneration2/E-code/assets/95161813/e98eff5e-b891-4953-bb71-44aacbf39fa5)


## abstract
Context: With the waning of Moore's Law, the software industry is placing increasing importance on finding alternative solutions for continuous performance enhancement. The significance and research results of software performance optimization have been on the rise in recent years, especially with the advancement propelled by \textbf{L}arge \textbf{L}anguage \textbf{M}odel\textbf{s} (LLMs). However, traditional strategies for rectifying performance flaws have shown significant limitations at the competitive code efficiency optimization level, and research on this topic is surprisingly scarce.

Objective: This study aims to address the research gap in this domain, offering practical solutions to the various challenges encountered. Specifically, we have overcome the constraints of traditional performance error rectification strategies and developed a Language Model (LM) tailored for the competitive code efficiency optimization realm.

Method: We introduced E-code, an advanced program synthesis LM. Inspired by the recent success of expert LMs, we designed an innovative structure called the Expert Encoder Group. This structure employs multiple expert encoders to extract features tailored for different input types. We assessed the performance of E-code against other leading models on a competitive dataset and conducted in-depth ablation experiments.

Results: Upon systematic evaluation, E-code achieved a 57% improvement in code efficiency, significantly outperforming other advanced models. In the ablation experiments, we further validated the significance of the expert encoder group and other components within E-code.

Conclusion: The research findings indicate that the expert encoder group can effectively handle various inputs in efficiency optimization tasks, significantly enhancing the model's performance. In summary, this study paves new avenues for developing systems and methods to assist programmers in writing efficient code.


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


#### Extremely fast use of Time_Predictor source code: 
1. Extract the GEC dataset to the E_code folder and change the file name to GEC. 
2. Run the train.py file to train the model.

3. Put the code to be predicted into Code_to_be_predicted a
4. Run Prediction_generation_code to automatically predict the code runtime.


## Model parameters
All model parameters are [here](https://drive.google.com/drive/folders/18tg9mTBZ3E6bmpnoelMbYqMo_o3B76bX?usp=sharing).

## CodeExecTimeDB
CodeExecTimeDB are [here](https://drive.google.com/file/d/1tR3R9Mf9thXBUszMo34Pmdli0K4savjp/view?usp=sharing).
