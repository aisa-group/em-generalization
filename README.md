# What Shapes Emergent Misalignment: Improving the Understanding of EM from the Perspective of Generalization

**Yuchen Zhang, Anietta Weckauff, Diego Garcia-Olano, Maksym Andriushchenko**  
ELLIS Institute Tübingen · Max Planck Institute for Intelligent Systems · Tübingen AI Center

### SFT Datasets
[data.zip](emergent-misalignment/data.zip) - contains the full dataset, password-locked with password `em`.
All training data do not contain system prompts. We use default system prompts from the tokenizers of the models.
We did not use all of the training data in this folder.

### Train and evals
Code and results are in [emergent-misalignment](emergent-misalignment/). This roughly follows the original [EM repo](https://github.com/emergent-misalignment/emergent-misalignment/) structure with some small fixes on eval.
Some large files (eval results) are excluded but can share upon request.

### Activation analysis
Code and results are in [activation_analysis](activation_analysis/). Activations are not included due to large file. 
* [get_activations](activation_analysis/get_activations): contains code to obtain activations 
* [analysis](activation_analysis/analysis): contains code for model prior eval activations predicting post narrow funetuning harmlessness level.
* [pca](activation_analysis/pca): contains code to fit pca and save the directions, project onto these directions and save
* [prompt_direction_change](activation_analysis/prompt_direction_change): contains code that compare the deltas of train and eval prompts before and after narrow finetuning (data element/part 3 of the paper).