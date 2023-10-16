

# Efficient-Code-Generation-with-E-code

![E-code模型图](https://github.com/CodeGeneration2/E-code/assets/95161813/e98eff5e-b891-4953-bb71-44aacbf39fa5)



The main processes of the E-code model, which we have labeled with numerical sequences (e.g., \textcircled{1}\textcircled{2}\textcircled{3}) in \ref{fig.E-code}, are as follows:
\begin{itemize}
\item \textbf{Step 1: Partitioning NL descriptions.} We divide the NL description into five parts: algorithm tags, problem descriptions, I/O format descriptions, and I/O test samples. Then, we input them into the expert encoder group. We describe our expert encoder group in detail in \ref{sec:5.2}. Its normalized expression is:
\begin{equation}
nl_t,nl_q,nl_i,nl_o,nl_s=f^{(Split)}(\text{NL input})
\end{equation}
where $f^{(Split)}$ is the segmentation function, $nl_t,nl_q,nl_i,nl_o,nl_s$ is the NL description of the segmentation into five parts.

\item \textbf{Step 2: Expert group processing.} The expert encoder group consists of five Bert-tiny models \cite{turc2019well,bhargava2021generalization}, each Bert-tiny model dealing with a separate section. Since the Bert-tiny models are small, they fit into our expert encoder group. The feature information set $e_X$ extracted by the expert group is computed by
\begin{equation}
e_{X} = f_{X}^{(Bert)}(nl_{X}),\text{ } X\in  \{t,q,i,o,s\}
\end{equation}
where $f_{X}^{(Bert)}$ is the set of expert group functions.

\item \textbf{Step 3: Expert group integration layer integration.} The output of the expert encoder group is concatenated and integrated using Bert-tiny models \cite{turc2019well,bhargava2021generalization} so that split NL descriptions can focus on each other and extract deep-level features. These features are then fed into a layer of \textbf{M}ulti-\textbf{L}ayer \textbf{P}erceptron (MLP) layers for integration. Note that our MLP layer deliberately removes the ReLU activation function. We describe this in detail in \ref{sec:5.3}. The NL features $Enc^{(nl)}$ of the encoder is computed by
\begin{equation}
Enc^{(nl)} = W^{(Enl)} [f_{(Int)}^{(Bert)} concat(e_t , e_q , e_i , e_o , e_s)]
\end{equation}
where $concat$ is the operator of tensor concatenation, $f_{(Int)}^{(Bert)}$ is the expert group integration layer function, $W^{(Enl)}$ is the MLP enlarge layer weights.

\item \textbf{Step 4: Extracting information about inefficient code features.} The GPT-Neo model \cite{black2021gpt} is used to extract feature information from the inefficient code. Because the GPT-Neo model has been pre-trained for the code, we can reduce the computational overhead of fine-tuning again. Inefficient code features information $Enc^{(ic)}$ computed by
\begin{equation}
Enc^{(ic)} =f_{ic}^{(GPT-Neo)}(\text{IC input})
\end{equation}
where $f_{ic}^{(GPT-Neo)}$ is the GPT-Neo model for extracting information about inefficient code features.

\item \textbf{Step 5: Integrated encoder output.} Concatenate the features obtained by the expert group with the components extracted from the inefficient code as the output of the encoder.  The encoder output $Enc$ is computed by
\begin{equation}
Enc= concat(Enc^{(nl)},Enc^{(ic)})
\end{equation}

\item \textbf{Step 6: Extracting information about generated efficient code features.} Feature information is extracted from the generated efficient code using the GPT-Neo model.  The extracted efficient code feature information $Dec^{(ec)}$ is computed by
\begin{equation}
Dec^{(ec)} =f_{ec}^{(GPT-Neo)}(\text{EC input})
\end{equation}
where $f_{ec}^{(GPT-Neo)}$ is the GPT-Neo model for extracting information about efficient code features.


\item \textbf{Step 7: Multi-headed attention mechanism.} Integrate the feature information of the encoder output and the generated partially efficient code using the multi-head attention layer and feed it to the final output module. Since the model decoder is unique, the attention mechanism layers in the model are independent. All our attention layers use 48 headers to maximize the fusion of feature information. We explain in detail the reasons for using 48 headers in Section \ref{sec:5.3}. The multi-head attention layer output $Dec^{(multi-h)}$ is computed by
\begin{equation}
Dec^{(multi-h)} = concat(head_1,\dots ,head_H ) W_h
\end{equation}
where $H$=48 denotes the number of heads, $W_h$ is the weight. An attention layer is applied in each head $head_t$, computed by
\begin{equation}
head_t=softmax(QK^\top/{\sqrt{d_k}} )V
\end{equation}
where $d_k = d/H$ denotes the length of each features vector. $Q$, $K$ and $V$ are computed by
\begin{equation}
[Q,K,V]=[Dec^{(ec)},Enc,Enc]^\top  [W_Q ,W_k ,W_V]
\end{equation}
where $W_Q ,W_k ,W_V$ are model parameters.

\item \textbf{Step 8: Predicted output.} The GPT-Neo model \cite{black2021gpt} is again used as the final output module to predict the next token based on all the feature information. The probability of predicting the next token $p(t_i)$ is computed by
\begin{equation}
p(t_i)=softmax(f_{output}^{(GPT-Neo)}(Dec^{(multi-h)}))
\end{equation}
where $f_{output}^{(GPT-Neo)}$ is the GPT-Neo model with MLP layers.
\end{itemize}






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

## Diagrammatic figure
In the Efficient-Code-Generation-with-E-code work, the diagrammatic figure is in the [Diagrammatic figure folder](https://github.com/CodeGeneration2/Diagrammatic-figure/tree/main/Diagrammatic%20figure).



