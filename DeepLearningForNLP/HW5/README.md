# Transformer language model homework

This homework has three parts.

1. In the first part of the homework (`01_multi_head_attention.ipynb`), you will implement the key concept of Transformer: multi-head attention.
1. In the second part, you need to use your multi-head attention implementation and `torch.nn` layers to implement Transformer Encoder. Your starter code with detailed task explanations is located in python file (not a notebook) `transformer_lm/modeling_transformer.py`. After implementing the model, you can test it in `02_transformer.ipynb`.
2. Implement training loop in `03_training.ipynb` and train 10 character-level language models.

## When do I need a GPU?

Only for the third part of the homework. Parts 1 and 2 do not involve training and can be done even a laptop CPU.

## Setting up the environment

All required libraries are listed in `requirements.txt`. You can install them using `pip install -r requirements.txt`.

Feel free to use Colab or Jupyter Lab for the first part of the homework, but we stongly recommend to use a code editor like VSCode or PyCharm for the second part, as it involves more interaction with `.py` files. Here is a good tutorial on how to [setup VSCode for Python](https://www.youtube.com/watch?v=Z3i04RoI9Fk). Both of them also support jupyter notebooks, you just need to specify which jupyter kernel you want to use (most probably its `nlp_class`). For VSCode you may want to additionally install a [markdown extention](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one) to render files like this README.md.

**Monitor your model performance while it is training** in WadnB. This is the main purpose of this tool. If the model is not improving at all or is diverging, look up our "my model does not converge" checklist from Lecture 3. At the same time, if you get a very high test accuracy, your model might be cheating and your causal masking or data preprocessing is not implemented correctly. To help you understand what correct training loss plot and eval perplixity/accuracy should look like, we will post a couple of images in Slack. Your runs will most probaly look different, because of different hyperparemeters, but they should not be extremely different.

## Submitting this homework

> NOTE: Do not add `model.pt`, `./wandb` or other large files to your submission.

1. Restart your `.ipynb` notebooks (part 1, part 2, and interact) and reexecute them top-to-bottom via "Restart and run all" button.
Not executed notebooks or the notebooks with the cells not executed in order will receive 0 points.
2. Add your report PDF to the homework contents.
3. Delete `wandb` diretory it it was created. Zip this directory and submit the archive it to the Blackaord.
4. Submit a link to your wandb project (that has all training/eval results for all trained models) for this homewrok (name it "transformer_lm" or something like this) to the Blackboard.

## Notes Mark 
19 different tods's
Notebook 1 
-Completed 1.1 and 1.2 with test cases passing 1 
-Completed inline question 2 and not inline question1 , Review these as I answered Q2 without the context at the top 
- Moved selfAttenion class to modeling_attention.py , May be some imports issue that need to be resolved
- Majority of notebook 1 is complete bar some of the inline questions review again before submisson to ensure all questions are answered properly and that shapes are mentioned throughout 

## Notes Mark - Transcribe into hardback
-2.1 self.q,k,v(x) transforms from [bs,seq,input_size] into [bs,seq,hidden] because self.q is a linear obeject of shape (input_size,hidden)
- the view operartions transform hidden into num_heads heads each with head_size(self.head_size) 
- The transpose (1,2) operation move num_heads dimension to axis 1 preparing for batch wise computation 
- The reshape operation combines batch_size and num_heads into a single dimension 
- scores computes attention scores by appluying the formula Q*K^T/ scale

-2.2 torch.triu created an upper traingular matrix of seq^2 ensuring for each postion i attetnion can not see postions j > i 
- masked_fill_ replaces the elements of scores where casual mask with '-inf'  ensuring these postions get zero weight after softmax is applied 

- 2.3 Applies softmax with z as scores , dropout applies regulartixation 
- reshape splits bs * num_heads back into seperate dimensions , transpose moves seq to axis 1 for final output , the final reshape operation combines num_heads * head_size into hidden 
- self.mix processes the concatenated heads

- Notebook 2 
- Go back and take notes on all functions , understanding completly what they are doing 